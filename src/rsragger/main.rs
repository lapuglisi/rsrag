#![allow(unused)]

use async_stream::stream;
use futures_util::{
  StreamExt,
  stream::{self, Stream},
};
use pdf_oxide::{PdfDocument, converters::ConversionOptions};
use qdrant_client::{
  Payload, Qdrant,
  qdrant::{DenseVector, Document, PointStruct, UpsertPointsBuilder, Vector, Vectors},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{
  any::{self, Any, TypeId},
  collections::HashMap,
  env,
  error::Error,
  pin,
};

const DEFAULT_EMBED_SERVER: &str = "http://192.120.100.20:8888";
const DEFAULT_QDRANT_SERVER: &str = "http://192.120.100.20:6334";
const DEFAULT_CHUNK_DELIMS: &str = "\n.?!";

struct RaggerEngine {
  embed_server: String,
  qdrant_server: String,
  source: Option<String>,
  chunk_size: usize,
  delimiters: String,
  pdf_pw: Option<String>,
}

//
// Llama Embedding Response
//
#[derive(Deserialize, Debug)]
struct EmbedResponseData {
  index: u32,
  object: String,
  embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
struct EmbedResponseUsage {
  prompt_tokens: u32,
  total_tokens: u32,
}

#[derive(Deserialize, Debug)]
struct EmbedResponse {
  model: String,
  object: String,
  usage: EmbedResponseUsage,
  data: Vec<EmbedResponseData>,
}
////////////////////////////

#[derive(Deserialize, Serialize, Debug)]
struct EmbedRequest {
  model: Option<String>,
  input: String,
}

// Qdrant stuff
#[derive(Debug)]
struct QdrantUpserter {
  collection: String,
  vectors: Option<std::collections::HashMap<String, Vector>>,
  payload: HashMap<String, serde_json::Value>,
  embeddings: Option<Vec<f32>>,
}

impl QdrantUpserter {
  fn new(collection: &str) -> Self {
    Self {
      collection: String::from(collection),
      vectors: None,
      embeddings: None,
      payload: HashMap::new(),
    }
  }

  fn with_payload(mut self, key: &str, value: &str) -> Self {
    self.payload.insert(
      key.to_string(),
      serde_json::Value::String(String::from(value)),
    );
    self
  }

  fn add_document(mut self, name: &str, text: &str, model: &str) -> Self {
    let mut vectors = self.vectors.unwrap_or(HashMap::new());
    let doc = Document::new(text, model);

    vectors.insert(name.to_string(), Vector::from(doc));

    self.vectors = Some(vectors);

    self
  }

  fn add_dense(mut self, name: &str, embeds: &Vec<f32>) -> Self {
    let mut vectors = self.vectors.unwrap_or(HashMap::new());

    vectors.insert(name.to_string(), Vector::new_dense(embeds.as_slice()));
    self.vectors = Some(vectors);

    self
  }

  fn add_vector(mut self, name: &str, embeddings: Vec<f32>) -> Self {
    let mut vectors = self.vectors.unwrap_or(HashMap::new());

    vectors.insert(name.to_string(), Vector::new(embeddings));
    self.vectors = Some(vectors);

    self
  }

  fn with_embeddings(mut self, e: Vec<f32>) -> Self {
    self.embeddings = Some(e);
    self
  }
}

impl RaggerEngine {
  fn new() -> Self {
    Self {
      embed_server: DEFAULT_EMBED_SERVER.to_string(),
      qdrant_server: DEFAULT_QDRANT_SERVER.to_string(),
      source: None,
      chunk_size: 4096,
      delimiters: String::from(DEFAULT_CHUNK_DELIMS),
      pdf_pw: None,
    }
  }

  fn with_embed_server(mut self, e: String) -> Self {
    self.embed_server = e;
    self
  }

  fn with_qdrant_server(mut self, q: String) -> Self {
    self.qdrant_server = q;
    self
  }

  fn read_file(&self, file: std::path::PathBuf) -> Result<Vec<String>, Box<dyn Error>> {
    let mime = mime_guess::from_path(&file).first_or_text_plain();
    let s = file.to_string_lossy();
    let mut data: Vec<String> = Vec::new();

    match mime {
      m if m == mime_guess::mime::APPLICATION_PDF => {
        println!("reading a pdf file: {}", s);
        let mut doc = pdf_oxide::PdfDocument::open(&file);
        if doc.is_err() {
          let err_str = format!("could not open pdf: {}", doc.unwrap_err());
          eprintln!("{}", err_str);
          return Err(err_str)?;
        }
        let mut doc = doc.unwrap();

        if let Some(pwd) = self.pdf_pw.clone() {
          eprint!("authenticating pdf with provided password... ");
          match doc.authenticate(pwd.as_bytes()) {
            Ok(true) => {
              eprintln!("\x1b[32mok!\x1b[0m");
            }
            Ok(false) => {
              eprintln!("\x1b[33mfailed!\x1b[0m");
            }
            Err(e) => {
              eprintln!("\x1b[31merror\x1b[0m: {}", e);
            }
          }
        }
        let pages = doc.page_count().unwrap_or(0);

        for page in 0..pages {
          match doc.extract_text(page) {
            Ok(t) => {
              data.push(t);
            }
            Err(e) => {
              eprintln!("could not get page #{} text: {}", page, e);
            }
          }
        }
      }
      _ => {
        println!("reading text file: {}", s);
        let buf = std::fs::read_to_string(file)?;
        data.push(buf);
      }
    }

    Ok(data)
  }

  async fn get_embeddings(
    &self,
    input: &str,
    model: Option<&str>,
  ) -> Result<EmbedResponse, Box<dyn Error>> {
    let url = format!("{}/v1/embeddings", self.embed_server);

    let er = EmbedRequest {
      model: model.map(|s| s.to_string()),
      input: String::from(input),
    };

    let json = serde_json::to_string(&er)?;
    let client = Client::new().post(url).body(json).send().await?;

    let body = client.text().await?;
    match serde_json::from_str(&body) {
      Ok(r) => Ok(r),
      Err(e) => {
        eprintln!("embeddings error: {} {}", e, body);
        Err(e)?
      }
    }
  }

  async fn save_to_qdrant(&self, data: QdrantUpserter) -> Result<(), Box<dyn Error>> {
    let qdrant = Qdrant::from_url(&self.qdrant_server).build()?;

    let point_id = uuid::Uuid::now_v7().to_string();
    let payload: Payload = data.payload.into();

    let mut points: Vec<PointStruct> = Vec::new();

    if let Some(e) = data.embeddings {
      points.push(PointStruct::new(point_id, e, payload));
    } else if let Some(v) = data.vectors {
      points.push(PointStruct::new(point_id, v, payload));
    }

    println!("about to upsert {:?} into {}", points, data.collection);

    let up = UpsertPointsBuilder::new(data.collection, points);
    let resp = qdrant.upsert_points(up).await?;

    if resp.result.is_some() {
      let result = resp.result.unwrap();
      println!(
        "got qdrant result: [{}, {}]",
        result.operation_id.unwrap_or_default(),
        result.status
      );
    } else {
      println!("got qdrant response: {:?}", resp);
    }

    Ok(())
  }

  async fn do_the_chunk(&self, content: &str) -> impl Stream<Item = String> {
    stream! {
      let delims = self.delimiters.as_str();
      let bytes = content.as_bytes();
      let chunks: Vec<&[u8]> = chunk::chunk(bytes)
        .delimiters(delims.as_bytes())
        .size(self.chunk_size)
        .collect();

      let mut iter = chunks.iter();
      while let Some(c) = iter.next() {
        match str::from_utf8(c) {
          Ok(s) => {
            yield s.to_string();
          }
          Err(e) => {
            eprintln!("invalid UTF-8 buffer: {}", e);
          }
        }
      }
    }
  }
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  let mut args = env::args();
  let mut engine: RaggerEngine = RaggerEngine::new();
  let mut qdrant_collection: Option<String> = None;

  let mut iter = args.into_iter();
  while let Some(arg) = iter.next() {
    match arg.as_str() {
      "--qdrant" | "-q" => {
        if let Some(q) = iter.next() {
          engine = engine.with_qdrant_server(q);
        }
      }
      "--embed" | "-e" => {
        if let Some(e) = iter.next() {
          engine = engine.with_embed_server(e);
        }
      }
      "--source" | "-s" => {
        if let Some(s) = iter.next() {
          engine.source = Some(s);
        }
      }
      "--chunk" => {
        if let Some(c) = iter.next() {
          if let Ok(i) = c.parse::<usize>() {
            engine.chunk_size = i;
          }
        }
      }
      "--delims" | "--delimiters" => {
        if let Some(d) = iter.next() {
          engine.delimiters.clone_from(&d);
        }
      }
      "--pdfpw" => {
        if let Some(p) = iter.next() {
          engine.pdf_pw = Some(p);
        }
      }
      "--qdrant-collection" | "-qc" => {
        if let Some(c) = iter.next() {
          qdrant_collection = Some(c);
        }
      }
      _ => {}
    }
  }

  println!();
  println!("using:");
  println!("engine.embed_server ...... {}", engine.embed_server);
  println!("engine.qdrant_server ..... {}", engine.qdrant_server);
  println!("engine.source ............ {:?}", engine.source);
  println!("engine.chunk_size ........ {}", engine.chunk_size);
  println!("engine.delimiters ........ {}", engine.delimiters);
  println!("engine.pdf_pw ............ {:?}", engine.pdf_pw);
  println!("qdrant_collection ........ {:?}", qdrant_collection);
  println!();

  if engine.source.is_none() {
    Err("--source is mandatory.")?
  }

  let source = engine.source.to_owned().unwrap();
  let mut content: Vec<String> = Vec::new();

  if std::fs::exists(&source).unwrap_or(false) {
    content = engine.read_file(std::path::PathBuf::from(&source))?;
  } else {
    content.push(source.clone());
  }

  // TODO: check endpoints health
  for input in content {
    let chunks = engine.do_the_chunk(&input).await;
    tokio::pin!(chunks);

    while let Some(chunk) = chunks.next().await {
      // get embeddings for each used model (embeddings and colbert)
      // and upsert them accordingly
      //
      //
      if let Some(collection) = qdrant_collection.as_ref() {
        let mut qu = QdrantUpserter::new(collection)
          .with_payload("document", &source)
          .with_payload("source", &chunk);

        let mut model = "embeddings:default";
        println!("getting embeddings from '{}' for '{:?}'...", model, chunk);
        let er = engine.get_embeddings(&chunk, Some(model)).await?;

        qu = qu.add_dense("text-dense", &er.data[0].embedding);

        model = "embeddings:colbert";
        println!("getting embeddings from '{}' for '{:?}'...", model, chunk);

        let er = engine.get_embeddings(&chunk, Some(model)).await?;
        qu = qu.add_dense("text-colbert", &er.data[0].embedding);
        qu = qu.add_document("text-sparse", &chunk, "qdrant/bm25");

        // got it, update embeddings for
        //
        engine.save_to_qdrant(qu).await?;
      } else {
        let er = engine.get_embeddings(&chunk, None).await?;

        let collection: String = er
          .model
          .chars()
          .map(|c| if c.is_ascii_punctuation() { '-' } else { c })
          .collect();

        for item in er.data.iter() {
          let qu = QdrantUpserter::new(&collection)
            .with_embeddings(item.embedding.clone())
            .with_payload("document", &source)
            .with_payload("source", &chunk);

          engine.save_to_qdrant(qu).await?;
        }
      }
    }
  }

  Ok(())
}
