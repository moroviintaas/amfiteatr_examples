mod prisoner;
mod historyless;


use std::fmt::{Debug, Formatter};
use amfi::agent::{AgentIdentifier, InformationSet};
use amfi::domain::DomainParameters;
pub use prisoner::*;
pub use historyless::*;
