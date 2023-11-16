use std::cmp::Ordering;
use std::marker::PhantomData;
use rand::distributions::Uniform;
use rand::prelude::ThreadRng;
use rand::{Rng, thread_rng};
use amfi::agent::{InformationSet, Policy};
use crate::classic::domain::{ClassicAction, ClassicGameDomain, ClassicGameError};
use crate::classic::domain::ClassicAction::{Cooperate, Defect};

pub struct ClassicPureStrategy<IS: InformationSet<ClassicGameDomain>>{
    pub action: ClassicAction,
    _is: PhantomData<IS>
}

impl<IS: InformationSet<ClassicGameDomain>> ClassicPureStrategy<IS>{
    pub fn new(action: ClassicAction) -> Self{
        Self{
            action,
            _is: Default::default()
        }
    }



}
impl<IS: InformationSet<ClassicGameDomain>> Policy<ClassicGameDomain> for ClassicPureStrategy<IS>{
    type InfoSetType = IS ;

    fn select_action(&self, _state: &Self::InfoSetType) -> Option<ClassicAction> {
        Some(self.action)
    }
}

pub struct ClassicMixedStrategy<IS: InformationSet<ClassicGameDomain>>{
    probability_defect: f64,
    _is: PhantomData<IS>
}

impl<IS: InformationSet<ClassicGameDomain>> ClassicMixedStrategy<IS>{
    pub fn new(probability_defect: f64) -> Self{
        Self{
            probability_defect,
            _is: Default::default(),
        }
    }
    pub fn new_checked(probability: f64) -> Result<Self, ClassicGameError>{
        if probability < 0.0 || probability > 1.0{
            Err(ClassicGameError::NotAProbability(probability))
        } else{
            Ok(Self::new(probability))
        }
    }
}

impl<IS: InformationSet<ClassicGameDomain>> Policy<ClassicGameDomain> for ClassicMixedStrategy<IS>{
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

