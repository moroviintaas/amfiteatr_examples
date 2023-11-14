use std::collections::HashMap;
use std::{thread};
use std::path::PathBuf;
use clap::{Parser, ValueEnum};
use log::LevelFilter;
use tch::{Device, nn, Tensor};
use tch::nn::{Adam, VarStore};
use amfi::agent::{*};
use amfi::comm::{SyncCommAgent, SyncCommEnv};
use amfi::env::generic::HashMapEnvT;
use amfi::env::{ReinitEnvironment, RoundRobinUniversalEnvironment};
use amfi::error::AmfiError;
use amfi_examples::prisoner::agent::{*};
use amfi_examples::prisoner::common::RewardTable;
use amfi_examples::prisoner::domain::PrisonerDomain;
use amfi_examples::prisoner::domain::PrisonerId::{Andrzej, Janusz};
use amfi_examples::prisoner::env::PrisonerEnvState;
use amfi_rl::actor_critic::ActorCriticPolicy;
use amfi_rl::{LearningNetworkPolicy, TrainConfig};
use amfi_rl::torch_net::{A2CNet, TensorA2C};



#[derive(ValueEnum, Debug, Copy, Clone)]
pub enum RewardSource{
    Env,
    State
}

#[derive(Parser)]
#[command(author, version, about, long_about = None)]
struct ExampleOptions{

    #[arg(short = 'v', long = "log_level", value_enum, default_value = "OFF")]
    pub log_level: LevelFilter,

    #[arg(short = 'o', long = "logfile")]
    pub log_file: Option<PathBuf>,

    #[arg(short = 's', long = "save")]
    pub save_file: Option<PathBuf>,

    #[arg(short = 'l', long = "load")]
    pub load_file: Option<PathBuf>,

    #[arg(short = 'e', long = "epochs", default_value = "10")]
    pub epochs: usize,

    #[arg(short = 'r', long = "reward", default_value = "env")]
    pub reward_source: RewardSource,

}

pub fn setup_logger(log_level: LevelFilter, log_file: &Option<PathBuf>) -> Result<(), fern::InitError> {
    let dispatch  = fern::Dispatch::new()

        .format(|out, message, record| {
            out.finish(format_args!(
                "{}[{}][{}] {}",
                chrono::Local::now().format("[%Y-%m-%d][%H:%M:%S]"),
                record.target(),
                record.level(),
                message
            ))
        })
        .level(log_level);

        match log_file{
            None => dispatch.chain(std::io::stdout()),
            Some(f) => dispatch.chain(fern::log_file(f)?)
        }

        .apply()?;
    Ok(())
}



struct PrisonerModel<P0: Policy<PrisonerDomain, InfoSetType=PrisonerInfoSet>, P1: Policy<PrisonerDomain, InfoSetType=PrisonerInfoSet>>{
    pub env: HashMapEnvT<PrisonerDomain, PrisonerEnvState, SyncCommEnv<PrisonerDomain>>,
    pub agent0: AgentGenT<PrisonerDomain, P0, SyncCommAgent<PrisonerDomain>>,
    pub agent1: AgentGenT<PrisonerDomain, P1, SyncCommAgent<PrisonerDomain>>,
    pub env_default_state: PrisonerEnvState,
    pub agent0_default_state: <P0 as Policy<PrisonerDomain>>::InfoSetType,
    pub agent1_default_state: <P1 as Policy<PrisonerDomain>>::InfoSetType,

}

impl <P0: Policy<PrisonerDomain, InfoSetType=PrisonerInfoSet>, P1: Policy<PrisonerDomain, InfoSetType=PrisonerInfoSet>> PrisonerModel<P0, P1>{

    pub fn evaluate(&mut self, number_of_tries: usize) -> Result<((f64, f64), (f64, f64)), AmfiError<PrisonerDomain>>{
        let mut sum_rewards_0_uni = 0.0;
        let mut sum_rewards_1_uni = 0.0;
        let mut sum_rewards_0_sub = 0.0;
        let mut sum_rewards_1_sub = 0.0;


        for _ in 0..number_of_tries{
            self.env.reinit(self.env_default_state.clone());
            self.agent0.reinit(self.agent0_default_state.clone());
            self.agent1.reinit(self.agent1_default_state.clone());
            thread::scope(|s|{
                s.spawn(||{
                    self.env.run_round_robin_uni_rewards().unwrap();
                });
                s.spawn(||{
                    self.agent0.run_rewarded().unwrap();
                });
                s.spawn(||{
                    self.agent1.run_rewarded().unwrap();
                });

            });

            sum_rewards_0_uni += self.agent0.current_universal_score() as f64;
            sum_rewards_1_uni += self.agent1.current_universal_score() as f64;
            sum_rewards_0_sub += self.agent0.current_subjective_score();
            sum_rewards_1_sub += self.agent1.current_subjective_score();
        }

        sum_rewards_0_uni /= number_of_tries as f64;
        sum_rewards_1_uni /= number_of_tries as f64;
        sum_rewards_0_sub /= number_of_tries as f64;
        sum_rewards_1_sub /= number_of_tries as f64;


        Ok(((sum_rewards_0_uni, sum_rewards_0_sub), (sum_rewards_1_uni, sum_rewards_1_sub)))
    }


}

impl<
    P0: Policy<PrisonerDomain,
        InfoSetType=PrisonerInfoSet>
> PrisonerModel<
    P0,
    ActorCriticPolicy<
        PrisonerDomain,
        PrisonerInfoSet,
        //PrisonerStateTranslate
        PrisonerInfoSetWay
    >
>{

    fn train_agent_1(&mut self, epochs: usize, games_in_epoch: usize, reward_source: RewardSource) -> Result<(), AmfiError<PrisonerDomain>>{

        let mut trajectory_archive = Vec::with_capacity(games_in_epoch);
        for epoch in 0..epochs{
            trajectory_archive.clear();
            for _game in 0..games_in_epoch{
                self.agent0.reinit(self.agent0_default_state.clone());
                self.agent1.reinit(self.agent1_default_state.clone());
                self.env.reinit(self.env_default_state.clone());

                thread::scope(|s|{
                    s.spawn(||{
                        self.env.run_round_robin_uni_rewards().unwrap();
                    });
                    s.spawn(||{
                        self.agent0.run_rewarded().unwrap();
                    });
                    s.spawn(||{
                        self.agent1.run_rewarded().unwrap();
                    });

                });

                trajectory_archive.push(self.agent1.take_trajectory());

            }

            match reward_source{
                RewardSource::Env => self.agent1.policy_mut().train_on_trajectories_env_reward(&trajectory_archive[..], 0.99).unwrap(),
                RewardSource::State => self.agent1.policy_mut().train_on_trajectories_info_set_rewards(&trajectory_archive[..], 0.99).unwrap(),
            };

            let scores = self.evaluate(1000)?;
            println!("Epoch {}: agent 0: ({} | {:.2}); agent 1: ({} | {:.2})", epoch, scores.0.0, scores.0.1, scores.1.0, scores.1.1);
            trajectory_archive.clear();

        }
        Ok(())
    }

}

fn main() -> Result<(), AmfiError<PrisonerDomain>>{
    let device = Device::Cpu;

    let args = ExampleOptions::parse();

    //setup_logger(LevelFilter::Debug, &None).unwrap();
    setup_logger(args.log_level, &args.log_file).unwrap();

    let reward_table = RewardTable{
        cover_v_cover: 5,
        betray_v_cover: 10,
        betray_v_betray: 3,
        cover_v_betray: 1
    };

    let initial_env_state = PrisonerEnvState::new(reward_table, 10);
    let env_state = initial_env_state.clone();

    let (comm_env_0, comm_prisoner_0) = SyncCommEnv::new_pair();
    let (comm_env_1, comm_prisoner_1) = SyncCommEnv::new_pair();

    let initial_prisoner_state = PrisonerInfoSet::new(reward_table);

    let prisoner0 = AgentGenT::new(
        Andrzej,
        PrisonerInfoSet::new(reward_table), comm_prisoner_0, SwitchOnTwoSubsequent{});



    let var_store = VarStore::new(device);
    let neural_net = A2CNet::new(var_store, |path|{
        let seq = nn::seq()
        .add(nn::linear(path / "input", 512, 1024, Default::default()))
        .add(nn::linear(path / "hidden", 1024, 1024, Default::default()));
        let actor = nn::linear(path / "al", 1024, 2, Default::default());
        let critic = nn::linear(path / "cl", 1024, 1, Default::default());
        let device = path.device();
        {move |xs: &Tensor|{
            let xs = xs.to_device(device).apply(&seq);
            //(xs.apply(&critic), xs.apply(&actor))
            TensorA2C{critic: xs.apply(&critic), actor: xs.apply(&actor)}
        }}
    });

    let optimiser = neural_net.build_optimizer(Adam::default(), 1e-4).unwrap();
    //let n_policy = ActorCriticPolicy::new(neural_net, optimiser, PrisonerStateTranslate {});
    let n_policy = ActorCriticPolicy::new(neural_net, optimiser, PrisonerInfoSetWay {}, TrainConfig { gamma: 0.99 });

    let mut prisoner1 = AgentGenT::new(
        Janusz,
        PrisonerInfoSet::new(reward_table), comm_prisoner_1, n_policy);

    if let Some(var_store_file) = args.load_file{
        prisoner1.policy_mut().network_mut().var_store_mut().load(var_store_file)
            .expect("Failed loading vars from file");
    }
    let mut env_coms = HashMap::new();
    env_coms.insert(Andrzej, comm_env_0);
    env_coms.insert(Janusz, comm_env_1);
    let env = HashMapEnvT::new(env_state, env_coms);


    let mut model = PrisonerModel{
        env, agent1: prisoner1, agent0: prisoner0, agent1_default_state: initial_prisoner_state.clone(),
        agent0_default_state: initial_prisoner_state.clone(), env_default_state: initial_env_state.clone()
    };

    let scores = model.evaluate(1000)?;
    println!("Before training: agent 0: ({} | {}); agent 1: ({} | {})", scores.0.0, scores.0.1, scores.1.0, scores.1.1);


    model.train_agent_1(args.epochs, 128, args.reward_source)?;

    if let Some(var_store_file) = args.save_file{
        model.agent1.policy().network().var_store().save(var_store_file)
            .expect("Failed saving vars to file");
    }

    /*


    let epochs = 1;
    let games_in_epoch = 2;
    let mut trajectory_archive = Vec::with_capacity(games_in_epoch);
    for epoch in 0..epochs{
        trajectory_archive.clear();
        for game in 0..games_in_epoch{
            prisoner0.reset(initial_prisoner_state.clone());
            prisoner1.reset(initial_prisoner_state.clone());
            env.reset(initial_env_state.clone());

            thread::scope(|s|{
                s.spawn(||{
                    env.run_round_robin_uni_rewards().unwrap();
                });
                s.spawn(||{
                    prisoner0.run_rewarded().unwrap();
                });
                s.spawn(||{
                    prisoner1.run_rewarded().unwrap();
                });

            });

            trajectory_archive.push(prisoner1.take_trajectory());

        }

        prisoner1.policy_mut().batch_train_env_rewards(&trajectory_archive[..], 0.99).unwrap();
        trajectory_archive.clear();

    }

     */

    Ok(())
}