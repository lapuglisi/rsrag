use crate::config::RagConfig;
use crate::engine::llama::LlamaEngine;

mod api;
mod config;
mod engine;

#[tokio::main]
async fn main() -> Result<(), Box<dyn std::error::Error>> {
  let llama: LlamaEngine = LlamaEngine::new();
  let settings: RagConfig = crate::config::RagConfig::load(None)?;

  println!("got settings: {:?}", settings);

  // then initialize llama

  // then initialize qdrant

  // then create the api object
  let api: api::RagApi = api::RagApi::new();

  // the do it
  api.listen().await;

  Ok(())
}
