#![allow(unused)]

use async_stream::stream;
use futures_util::{
  StreamExt,
  stream::{self, Stream},
};
use qdrant_client::{
  Payload, Qdrant,
  qdrant::{PointStruct, UpsertPoints, UpsertPointsBuilder},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{env, error::Error, pin};

const DEFAULT_EMBED_SERVER: &str = "http://192.120.100.20:8888";
const DEFAULT_QDRANT_SERVER: &str = "http://192.120.100.20:6334";
const DEFAULT_CHUNK_DELIMS: &str = ".!?\n";

struct RaggerEngine {
  embed_server: String,
  qdrant_server: String,
  source: Option<String>,
  chunk_size: usize,
  delimiters: String,
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
  input: String,
}

impl RaggerEngine {
  fn new() -> Self {
    Self {
      embed_server: DEFAULT_EMBED_SERVER.to_string(),
      qdrant_server: DEFAULT_QDRANT_SERVER.to_string(),
      source: None,
      chunk_size: 4096,
      delimiters: String::from(DEFAULT_CHUNK_DELIMS),
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
        data = pdf_extract::extract_text_by_pages(file)?;
      }
      _ => {
        println!("reading text file: {}", s);
        let buf = std::fs::read_to_string(file)?;
        data.push(buf);
      }
    }

    Ok(data)
  }

  async fn get_embeddings(&self, input: &str) -> Result<EmbedResponse, Box<dyn Error>> {
    let url = format!("{}/v1/embeddings", self.embed_server);

    let er = EmbedRequest {
      input: input.to_string(),
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

    let mut payload = Payload::new();
    payload.insert("source", data.source);
    payload.insert("document", data.document);

    let point = PointStruct::new(point_id, data.embeddings, payload);
    println!("about to upsert {:?} into {}", point.id, data.collection);

    let up = UpsertPointsBuilder::new(&data.collection, vec![point]).build();

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

// Qdrant stuff
#[derive(Debug)]
struct QdrantUpserter {
  collection: String,
  embeddings: Vec<f32>,
  document: String,
  source: String,
}

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  let mut args = env::args();
  let mut engine: RaggerEngine = RaggerEngine::new();

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
      _ => {}
    }
  }

  println!();
  println!("using:");
  println!("engine.embed_server ...... {}", engine.embed_server);
  println!("engine.qdrant_server ..... {}", engine.qdrant_server);
  println!("engine.source ............ {:?}", engine.source);
  println!("engine.chunk_size ........ {}", engine.chunk_size);
  println!("engine.delimiters ........ {:?}", engine.delimiters);
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
      println!("getting embeddings for '{:?}'...", chunk);
      let er = engine.get_embeddings(&chunk).await?;

      let collection: String = er
        .model
        .chars()
        .map(|c| if c.is_ascii_punctuation() { '-' } else { c })
        .collect();

      for item in er.data.iter() {
        let qu = QdrantUpserter {
          collection: collection.to_owned(),
          document: source.to_owned(),
          source: input.to_owned(),
          embeddings: item.embedding.clone(),
        };

        engine.save_to_qdrant(qu).await?;
      }
    }
  }

  Ok(())
}
