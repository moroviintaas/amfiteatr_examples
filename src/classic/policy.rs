use std::cmp::Ordering;
use std::marker::PhantomData;
use rand::distributions::Uniform;
use rand::prelude::ThreadRng;
use rand::{Rng, thread_rng};
use amfi::agent::{AgentIdentifier, InformationSet, Policy};
use crate::classic::domain::{ClassicAction, ClassicGameDomain, ClassicGameDomainNamed, ClassicGameError, PrisonerId};
use crate::classic::domain::ClassicAction::{Cooperate, Defect};

pub struct ClassicPureStrategy<ID: AgentIdentifier, IS: InformationSet<ClassicGameDomain<ID>>>{
    pub action: ClassicAction,
    _is: PhantomData<IS>,
    _id: PhantomData<ID>,
}

impl<ID: AgentIdentifier, IS: InformationSet<ClassicGameDomain<ID>>> ClassicPureStrategy<ID, IS>{
    pub fn new(action: ClassicAction) -> Self{
        Self{
            action,
            _is: Default::default(),
            _id: Default::default()
        }
    }



}
impl<ID: AgentIdentifier, IS: InformationSet<ClassicGameDomain<ID>>> Policy<ClassicGameDomain<ID>> for ClassicPureStrategy<ID, IS>{
    type InfoSetType = IS ;

    fn select_action(&self, _state: &Self::InfoSetType) -> Option<ClassicAction> {
        Some(self.action)
    }
}

pub struct ClassicMixedStrategy<ID: AgentIdentifier, IS: InformationSet<ClassicGameDomain<ID>>>{
    probability_defect: f64,
    _is: PhantomData<IS>,
    _id: PhantomData<ID>,
}

impl<ID: AgentIdentifier, IS: InformationSet<ClassicGameDomain<ID>>> ClassicMixedStrategy<ID, IS>{
    pub fn new(probability_defect: f64) -> Self{
        Self{
            probability_defect,
            _is: Default::default(),
            _id: Default::default(),
        }
    }
    pub fn new_checked(probability: f64) -> Result<Self, ClassicGameError<ID>>{
        if probability < 0.0 || probability > 1.0{
            Err(ClassicGameError::NotAProbability(probability))
        } else{
            Ok(Self::new(probability))
        }
    }
}

impl<ID: AgentIdentifier, IS: InformationSet<ClassicGameDomain<ID>>> Policy<ClassicGameDomain<ID>> for ClassicMixedStrategy<ID, IS>{
    type InfoSetType = IS ;

    fn select_action(&self, _state: &Self::InfoSetType) -> Option<ClassicAction> {
        let mut rng = thread_rng();
        let sample = rng.gen_range(0.0..1.0);
        sample.partial_cmp(&self.probability_defect).and_then(|o|{
          match o{
              Ordering::Less => Some(Defect),
              Ordering::Equal => Some(Cooperate),
              Ordering::Greater => Some(Cooperate),
          }
        })

    }
}

