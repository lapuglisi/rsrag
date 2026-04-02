#![allow(unused)]

use async_stream::stream;
use futures_util::stream::{self, Stream};
use qdrant_client::Payload;
use qdrant_client::config::QdrantConfig;
use qdrant_client::qdrant::points_update_operation::PointStructList;
use qdrant_client::qdrant::{PointStruct, UpsertPointsBuilder};
use qdrant_client::{Qdrant, qdrant::UpsertPoints};
use reqwest::Client;
use serde::{Deserialize, Serialize};
use std::error::Error;
use std::{env, io};

const DEFAULT_EMBED_SERVER: &str = "http://192.120.100.20:8888";
const DEFAULT_QDRANT_SERVER: &str = "http://192.120.100.20:6334";

struct RaggerEngine {
  embed_server: String,
  qdrant_server: String,
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

  async fn get_embeddings(&self, input: &str) -> Result<EmbedResponse, Box<dyn Error>> {
    let url = format!("{}/v1/embeddings", self.embed_server);

    let er = EmbedRequest {
      input: input.to_string(),
    };

    let json = serde_json::to_string(&er)?;
    let client = Client::new().post(url).body(json).send().await?;

    let body = client.text().await?;
    let resp: EmbedResponse = serde_json::from_str(&body)?;

    Ok(resp)
  }

  async fn save_to_qdrant(&self, data: QdrantUpserter) -> Result<(), Box<dyn Error>> {
    let qdrant = Qdrant::from_url(&self.qdrant_server).build()?;

    let point_id = uuid::Uuid::now_v7().to_string();

    let mut payload = Payload::new();
    payload.insert("source", data.source);
    payload.insert("document", data.document);

    let point = PointStruct::new(point_id, data.embeddings, payload);
    eprintln!("about to upsert {:?} into {}", point, data.collection);

    let up = UpsertPointsBuilder::new(&data.collection, vec![point]).build();

    let resp = qdrant.upsert_points(up).await?;

    eprintln!("got qdrant response: {:?}", resp);

    Ok(())
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
      _ => {}
    }
  }

  println!();
  println!("using:");
  println!("engine.embed_server ...... {}", engine.embed_server);
  println!("engine.qdrant_server ..... {}", engine.qdrant_server);
  println!();

  // TODO: check endpoints health
  let input = "Busco sexo com alegria e satisfação.";
  let er = engine.get_embeddings(input).await?;

  let collection: String = er
    .model
    .chars()
    .map(|c| if c.is_ascii_punctuation() { '-' } else { c })
    .collect();

  for item in er.data.iter() {
    let qu = QdrantUpserter {
      collection: collection.to_owned(),
      document: "busco".to_string(),
      source: "asdasdasdasdasd".to_string(),
      embeddings: item.embedding.clone(),
    };

    engine.save_to_qdrant(qu).await?;
  }

  Ok(())
}

// TODO: chunk document
async fn do_the_doc_chunk(doc: String) -> impl Stream<Item = Option<String>> {
  stream! {
    let file = std::fs::File::open(&doc);

    if file.is_err() {
      eprintln!("could not open file '{}': {}", doc, file.unwrap_err());
      yield None;
      return;
    }

    let file = file.unwrap();
  }
}
