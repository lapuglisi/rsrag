use async_stream::stream;
use futures_util::{StreamExt, stream::Stream};
use notify::Watcher;
use pdf_oxide::converters::ConversionOptions;
use qdrant_client::{
  Payload, Qdrant,
  qdrant::{Document, PointStruct, UpsertPointsBuilder, Vector},
};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::{collections::HashMap, env, error::Error};

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
  qdrant_collection: Option<String>,
}

//
// Llama Embedding Response
//
#[allow(dead_code)]
#[derive(Deserialize, Debug)]
struct EmbedResponseData {
  index: u32,
  object: String,
  embedding: Vec<f32>,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
struct EmbedResponseUsage {
  prompt_tokens: u32,
  total_tokens: u32,
}

#[derive(Deserialize, Debug)]
#[allow(dead_code)]
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

#[allow(dead_code)]
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
      qdrant_collection: None,
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
        log::info!("reading a pdf file: {}", s);
        let doc = pdf_oxide::PdfDocument::open(&file);
        if doc.is_err() {
          let err_str = format!("could not open pdf: {}", doc.unwrap_err());
          log::error!("{}", err_str);
          return Err(err_str)?;
        }
        let mut doc = doc.unwrap();

        if let Some(pwd) = self.pdf_pw.clone() {
          log::info!("authenticating pdf with provided password... ");
          match doc.authenticate(pwd.as_bytes()) {
            Ok(true) => {
              log::info!("authentication succeeded!");
            }
            Ok(false) => {
              log::warn!("authentication failed!");
            }
            Err(e) => {
              log::error!("authentication error: {}", e);
            }
          }
        }
        let pages = doc.page_count().unwrap_or(0);

        for page in 0..pages {
          let options = ConversionOptions {
            detect_headings: true,
            preserve_layout: true,
            include_images: false,
            extract_tables: true,
            ..Default::default()
          };
          match doc.to_markdown(page, &options) {
            Ok(t) => {
              data.push(t);
            }
            Err(e) => {
              log::error!("could not get page #{} text: {}", page, e);
            }
          }
        }
      }
      m if m.type_() == mime_guess::mime::TEXT => {
        log::info!("reading text file: {} ({})", s, m);
        let buf = std::fs::read_to_string(file)?;
        data.push(buf);
      }
      m => {
        let err = format!("unhandled mime type '{}'", m);
        log::warn!("{}: {}", s, err);
        return Err(err)?;
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
        log::error!("embeddings error: {} {}", e, body);
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

    log::info!("about to upsert {:?} into {}", points, data.collection);

    let up = UpsertPointsBuilder::new(data.collection, points);
    let resp = qdrant.upsert_points(up).await?;

    if resp.result.is_some() {
      let result = resp.result.unwrap();
      log::debug!(
        "got qdrant result: [{}, {}]",
        result.operation_id.unwrap_or_default(),
        result.status
      );
    } else {
      log::debug!("got qdrant response: {:?}", resp);
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
            log::error!("invalid UTF-8 buffer: {}", e);
          }
        }
      }
    }
  }
}

async fn handle_content(engine: &RaggerEngine, source: &str) -> Result<(), Box<dyn Error>> {
  let mut content: Vec<String> = Vec::new();

  if std::fs::exists(&source).unwrap_or(false) {
    content = engine.read_file(std::path::PathBuf::from(&source))?;
  } else {
    content.push(source.into());
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
      if let Some(collection) = engine.qdrant_collection.as_ref() {
        let mut qu = QdrantUpserter::new(collection)
          .with_payload("document", &source)
          .with_payload("source", &chunk);

        let mut model = "embeddings:default";
        log::info!("getting embeddings from '{}' for '{:?}'...", model, chunk);
        let er = engine.get_embeddings(&chunk, Some(model)).await?;

        qu = qu.add_dense("text-dense", &er.data[0].embedding);

        model = "embeddings:colbert";
        log::info!("getting embeddings from '{}' for '{:?}'...", model, chunk);

        let er = engine.get_embeddings(&chunk, Some(model)).await?;
        qu = qu.add_dense("text-colbert", &er.data[0].embedding);
        qu = qu.add_document("text-sparse", &chunk, "qdrant/bm25");

        // got it, update embeddings for
        //
        engine.save_to_qdrant(qu).await?;
      } else {
        let er = engine
          .get_embeddings(&chunk, Some("embeddings:default"))
          .await?;

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

#[tokio::main]
async fn main() -> Result<(), Box<dyn Error>> {
  let args = env::args();
  let mut engine: RaggerEngine = RaggerEngine::new();
  let mut log_file: Option<String> = None;
  let mut log_level: log::LevelFilter = log::LevelFilter::Info;
  let mut watch_dir: Option<String> = None;
  let mut watch_delete: bool = false;

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
        engine.source = iter.next();
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
        engine.pdf_pw = iter.next();
      }
      "--qdrant-collection" | "-qc" => {
        engine.qdrant_collection = iter.next();
      }
      "--log-file" | "--log" | "-lf" => log_file = iter.next(),
      "--debug" | "-d" => log_level = log::LevelFilter::Debug,
      "--watch" => watch_dir = iter.next(),
      "--watch-delete" | "-wd" => {
        if let Some(b) = iter.next() {
          watch_delete = b.parse::<bool>().unwrap_or(false);
        }
      }
      _ => {}
    }
  }

  // initialize log
  if log_file.is_some() {
    let log_file = log_file.unwrap();
    eprintln!("\x1b[34minfo\x1b[0m: logging to file '{}'", log_file);
    if let Err(e) = simple_logging::log_to_file(&log_file, log_level) {
      eprintln!(
        "\x1b[33mwarning\x1b[0m: could not log to file '{}' ({}). logging to stderr",
        log_file, e
      );
      simple_logging::log_to_stderr(log_level);
    }
  } else {
    eprintln!("\x1b[35minfo\x1b[0m: logging to stderr.");
    simple_logging::log_to_stderr(log_level);
  }

  log::info!("");
  log::info!("using:");
  log::info!("engine.embed_server ... {}", engine.embed_server);
  log::info!("engine.qdrant_server .. {}", engine.qdrant_server);
  log::info!("engine.source ......... {:?}", engine.source);
  log::info!("engine.chunk_size ..... {}", engine.chunk_size);
  log::info!("engine.delimiters ..... {}", engine.delimiters);
  log::info!("engine.pdf_pw ......... {:?}", engine.pdf_pw);
  log::info!("qdrant_collection ..... {:?}", engine.qdrant_collection);
  log::info!("watch_dir ............. {:?}", watch_dir);
  log::info!("watch_delete .......... {}", watch_delete);
  log::info!("");

  if engine.qdrant_collection.is_none() {
    log::warn!("\x1b[33mwarning\x1b[0m: Qdrant collection not defined!");
  }

  if watch_dir.is_some() {
    let watch_dir = watch_dir.unwrap();

    log::info!("reading files from {}", watch_dir);

    let (tx, rx) = std::sync::mpsc::channel::<notify::Result<notify::Event>>();

    log::info!("file watcher: starting...");
    let mut watcher = notify::recommended_watcher(tx)?;

    log::info!("file watcher: started.");

    watcher.watch(
      std::path::Path::new(&watch_dir),
      notify::RecursiveMode::Recursive,
    )?;

    for res in rx {
      match res {
        Ok(event) => {
          log::debug!("watcher: got event {:?}", event);

          if event.kind.is_create() || event.kind.is_modify() {
            log::info!("got 'created|modify' event: {:?}", event);

            for path in event.paths {
              let file = path.to_string_lossy();

              log::info!("handling path '{}'", file);
              if path.is_file() {
                log::info!("handling file '{}'", file);

                log::set_max_level(log::LevelFilter::Off);
                let call = handle_content(&engine, &file).await;
                log::set_max_level(log_level);

                if call.is_ok() {
                  log::info!("file '{}' handled successfully.", file);
                } else {
                  let err = call.unwrap_err();
                  log::error!("error while handling '{}': {}", file, err);
                }

                if watch_delete {
                  log::info!("removing file {}", file);

                  match std::fs::remove_file(&path) {
                    Ok(_) => log::info!("file {} removed", file),
                    Err(e) => log::error!("could not remove file {}: {}", file, e),
                  }
                } else {
                  log::info!("not deleting file {}", file);
                }
              } else {
                log::warn!("path '{}' is not a file. skipping", file);
              }
            }
          }
        }
        Err(e) => {
          log::error!("watcher error: {}", e);
        }
      }
    }

    Ok(())
  } else {
    if engine.source.is_none() {
      Err("--source is mandatory.")?
    }

    let source = engine.source.as_ref().unwrap();
    handle_content(&engine, source).await
  }
}
