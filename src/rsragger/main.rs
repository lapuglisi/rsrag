#![allow(unused)]

use async_stream::stream;
use axum::extract::Path;
use futures_util::stream::{self, Stream};
use std::error::Error;
use std::{env, io};

const DEFAULT_EMBED_SERVER: &str = "http://192.120.100.20:8888";
const DEFAULT_QDRANT_SERVER: &str = "http://192.120.100.20:6334";

struct RaggerEngine {
  embed_server: String,
  qdrant_server: String,
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
}

struct QdrantUpserter {
  embeddings: Vec<f32>,
  document: String,
  source: String,
}

fn main() -> Result<(), Box<dyn Error>> {
  // TODO: parse arguments
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

// TODO: request embeddings
fn get_embeddings(input: &str) -> Option<Vec<f32>> {
  None
}

// TODO: save embeddings to qdrant
fn save_to_qdrant(url: String, embed: Vec<f32>) -> Result<(), Box<dyn Error>> {
  Ok(())
}
