//! Session: orchestrates IPC, scene, assets, and frame flow.

#![allow(clippy::module_inception)]

pub mod commands;
pub mod init;
pub mod session;
pub mod state;

pub use commands::{CommandContext, CommandDispatcher, CommandResult};
pub use session::Session;
