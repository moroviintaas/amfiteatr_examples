use std::collections::HashMap;
use std::{option, thread};
use std::path::PathBuf;
use log::LevelFilter;
use tch::{Device, nn, Tensor};
use tch::nn::{Adam, VarStore};
use sztorm::agent::{AgentGenT, AutomaticAgentRewarded, CommunicatingAgent, PolicyAgent, ResetAgent, TracingAgent};
use sztorm::comm::SyncCommEnv;
use sztorm::env::generic::HashMapEnvT;
use sztorm::env::{ResetEnvironment, RoundRobinUniversalEnvironment};
use sztorm::error::SztormError;
use sztorm_examples::prisoner::agent::{CoverPolicy, PrisonerState, PrisonerStateTranslate, SwitchOnTwoSubsequent};
use sztorm_examples::prisoner::common::RewardTable;
use sztorm_examples::prisoner::domain::PrisonerDomain;
use sztorm_examples::prisoner::domain::PrisonerId::{Andrzej, Janusz};
use sztorm_examples::prisoner::env::PrisonerEnvState;
use sztorm_rl::actor_critic::ActorCriticPolicy;
use sztorm_rl::torch_net::{A2CNet, TensorA2C};

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

        //.chain(std::io::stdout())
        //.chain(fern::log_file("output.log")?)
        .apply()?;
    Ok(())
}
fn main() -> Result<(), SztormError<PrisonerDomain>>{
    let device = Device::cuda_if_available();

    setup_logger(LevelFilter::Debug, &None).unwrap();

    let reward_table = RewardTable{
        cover_v_cover: 5,
        betray_v_cover: 10,
        betray_v_betray: 3,
        cover_v_betray: 1
    };


    let env_state = PrisonerEnvState::new(reward_table,  10);

    let (comm_env_0, comm_prisoner_0) = SyncCommEnv::new_pair();
    let (comm_env_1, comm_prisoner_1) = SyncCommEnv::new_pair();

    let initial_prisoner_state = PrisonerState::new(reward_table);
    let initial_env_state = PrisonerEnvState::new(reward_table, 10);
    let mut prisoner0 = AgentGenT::new(
        Andrzej,
        PrisonerState::new(reward_table), comm_prisoner_0, SwitchOnTwoSubsequent{});

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
    let n_policy = ActorCriticPolicy::new(neural_net, optimiser, PrisonerStateTranslate {});

    let mut prisoner1 = AgentGenT::new(
        Janusz,
        PrisonerState::new(reward_table), comm_prisoner_1, n_policy);
    let mut env_coms = HashMap::new();
    env_coms.insert(Andrzej, comm_env_0);
    env_coms.insert(Janusz, comm_env_1);
    let mut env = HashMapEnvT::new(env_state, env_coms);

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

    Ok(())
}