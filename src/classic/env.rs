use std::fmt::{Display, Formatter};
use std::sync::Arc;
use enum_map::Enum;
use amfi::agent::AgentIdentifier;
use amfi::env::{EnvStateSequential, EnvironmentStateUniScore};
use amfi::domain::DomainParameters;
use crate::classic::common::{Side, SymmetricRewardTableInt};
use crate::classic::domain::{PRISONERS, ClassicAction, ClassicGameDomain, ClassicGameError, PrisonerId, EncounterReport, PrisonerMap, ClassicGameDomainNamed};
use crate::classic::domain::ClassicGameError::{ActionAfterGameOver, ActionOutOfOrder};
use crate::classic::domain::PrisonerId::{Andrzej, Janusz};


#[derive(Clone, Debug)]
pub struct PrisonerEnvState{
    previous_actions: Vec<PrisonerMap<ClassicAction>>,
    //last_actions: HashMap<PrisonerId, Option<PrisonerAction>>,
    last_round_actions: PrisonerMap<Option<ClassicAction>>,
    reward_table: SymmetricRewardTableInt,
    target_rounds: usize,

}

impl PrisonerEnvState{
    pub fn new(reward_table: SymmetricRewardTableInt, number_of_rounds: usize) -> Self{
        Self{
            previous_actions: Vec::with_capacity(number_of_rounds),
            last_round_actions: PrisonerMap::default(),
            reward_table,
            target_rounds: number_of_rounds
        }
    }
}


impl Display for PrisonerEnvState{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "({}/{}) | A-J: ", self.previous_actions.len(), self.target_rounds)?;
        for p in &self.previous_actions{
            //write!(f, "{},", p)?;
            /*
            match p[Andrzej]{
                PrisonerAction::Betray => write!(f, "B")?,
                PrisonerAction::Cover => write!(f, "C")?
            };
            match p[Janusz]{
                PrisonerAction::Betray => write!(f, "-B ")?,
                PrisonerAction::Cover => write!(f, "-C ")?
            };*/
            write!(f, "{:#}-{:#} ", p[Andrzej], p[Janusz])?;
        }
        write!(f, " | ")?;
        match self.last_round_actions[Andrzej]{
            None => write!(f, "N-")?,
            Some(s) => write!(f, "{:#}-", s)?
        };
        match self.last_round_actions[Janusz]{
            None => write!(f, "N")?,
            Some(s) => write!(f, "{:#}", s)?
        };
        write!(f, "")


    }
}

impl EnvStateSequential<ClassicGameDomainNamed> for PrisonerEnvState{
    type Updates = Vec<(PrisonerId, Arc<Vec<EncounterReport>>)>;

    fn current_player(&self) -> Option<PrisonerId> {
        if self.previous_actions.len() >= self.target_rounds{
            None
        } else{
            for i in PRISONERS{
                if self.last_round_actions[i].is_none(){
                    return Some(i)
                }
            }
            None
        }

    }





    fn is_finished(&self) -> bool {
        self.previous_actions.len() >= self.target_rounds
    }

    fn forward(&mut self, agent: PrisonerId, action: ClassicAction) -> Result<Self::Updates, ClassicGameError<PrisonerId>> {
        if self.is_finished(){
            return Err(ActionAfterGameOver(agent));
        }
        match self.last_round_actions[agent]{
            None => {
                self.last_round_actions[agent] = Some(action);


            },
            Some(_) => {
                return Err(ActionOutOfOrder(agent));
            }
        };
        for agent in PRISONERS {
            if self.last_round_actions[agent].is_none(){
                return Ok(Vec::default());
            }
        }

        //let a0 = self.last_round_actions[Andrzej].unwrap();
        //let a1 = self.last_round_actions[Janusz].unwrap();
        let a0 = self.last_round_actions[Andrzej].unwrap();
        let a1 = self.last_round_actions[Janusz].unwrap();
        let action_entry = PrisonerMap::new(a0, a1);
        self.previous_actions.push(action_entry);
        self.last_round_actions[Andrzej] = None;
        self.last_round_actions[Janusz] = None;
        /*
        let updates = vec![
            (Andrzej, EncounterReport {
                own_action: a0,
                other_player_action: a1,
                side: Side::Left,
                other_id: PrisonerId::Janusz,
            }),
            (Janusz, EncounterReport {
                own_action: a1,
                other_player_action: a0,
                side: Side::Right,
                other_id: PrisonerId::Andrzej
            })
        ];

         */

        let reports = Arc::new(vec![
            EncounterReport{
                own_action: action_entry[PrisonerId::from_usize(0)],
                other_player_action: action_entry[PrisonerId::from_usize(1)],
                side: Side::Left,
                other_id: PrisonerId::from_usize(1),
            },
            EncounterReport{
                own_action: action_entry[PrisonerId::from_usize(1)],
                other_player_action: action_entry[PrisonerId::from_usize(0)],
                side: Side::Right,
                other_id: PrisonerId::from_usize(0),
            },
        ]);
        let updates = vec![
            (PrisonerId::from_usize(0), reports.clone()),
            (PrisonerId::from_usize(1), reports),
        ];


        Ok(updates)

    }
}

impl EnvironmentStateUniScore<ClassicGameDomainNamed> for PrisonerEnvState{
    fn state_score_of_player(&self, agent: &PrisonerId) -> <ClassicGameDomainNamed as DomainParameters>::UniversalReward {
        let other = agent.other();
        self.previous_actions.iter().fold(0, |acc,x|{
            acc  + self.reward_table.reward(x[*agent], x[other])
        })
    }
}