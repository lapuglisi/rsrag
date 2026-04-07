use crate::engine::llama::LlamaEngine;
use crate::{config::RagConfig, engine::qdrant::QdrantEngine};

use log::LevelFilter;
use std::env;

mod api;
mod config;
mod engine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  let args: Vec<String> = env::args().collect();
  let mut config_file: Option<String> = None;
  let mut log_level: LevelFilter = LevelFilter::Info;

  let mut iter = args.iter();
  while let Some(arg) = iter.next() {
    println!("inside while: arg is {}", arg);
    match arg.as_str() {
      "--config" | "-c" => {
        config_file = iter.next().map(|a| a.to_string());
      }
      "--debug" | "-d" => {
        log_level = LevelFilter::Debug;
      }
      _ => {
        eprintln!("unknown arg: {}", arg);
      }
    }
  }

  let settings: RagConfig = crate::config::RagConfig::load(config_file)?;

  if settings.log_debug {
    println!("using Debug log_level defined in settings.");
    log_level = LevelFilter::Debug
  }

  if settings.log_file.is_some() {
    let log_file = settings.log_file.as_ref().unwrap();
    println!("using log file '{}'.", log_file);

    if let Err(_) = simple_logging::log_to_file(log_file, log_level) {
      eprintln!(
        "could not use log file '{}': logging to stderr instead.",
        log_file
      );
      simple_logging::log_to_stderr(log_level);
    }
  } else {
    eprintln!("no log file provided; logging to stderr");
    simple_logging::log_to_stderr(log_level);
  }

  log::info!("got settings: {:?}", settings);

  // then initialize llama
  log::debug!("initializing llama engine.");
  let llama = LlamaEngine::new()
    .with_llama_server(&settings.llama.chat_server)
    .with_embed_server(&settings.llama.embed_server)
    .with_rerank_server(&settings.llama.rerank_server);

  // then initialize qdrant
  let qdrant_url = format!("{}:{}", settings.qdrant.host, settings.qdrant.port);
  log::info!("initialize qdrant: {}", qdrant_url);

  let qdrant: QdrantEngine =
    QdrantEngine::new(qdrant_url).with_collection(&settings.qdrant.collection);

  // then create the api object
  let api = api::RagApi::new()
    .with_listen_host(&settings.http.host)
    .with_listen_port(settings.http.port)
    .with_llama_engine(llama)
    .with_qdrant_engine(qdrant);

  // the do it
  match api.listen().await {
    Ok(v) => Ok(v),
    Err(e) => Err(e),
  }
}
