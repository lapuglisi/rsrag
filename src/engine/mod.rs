use serde::Serialize;

pub mod anthropic;
pub mod llama;
pub mod qdrant;

#[derive(Serialize, Debug)]
pub struct AICompletionMessage {
  pub role: String,
  pub content: String,
}

impl Default for AICompletionMessage {
  fn default() -> Self {
    Self {
      role: String::new(),
      content: String::new(),
    }
  }
}

impl AICompletionMessage {
  pub fn new() -> Self {
    Self::default()
  }

  pub fn new_with_role(role: &str) -> Self {
    Self {
      role: role.to_string(),
      content: String::new(),
    }
  }

  pub fn with_role(mut self, role: &str) -> Self {
    self.role = role.to_string();
    self
  }

  pub fn with_content(mut self, content: &str) -> Self {
    self.content = content.to_string();
    self
  }
}

//
//
//
