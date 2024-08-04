import argparse, json, copy, os
import pickle
from deep_dialog.dialog_system import DialogManager, text_to_dict
from deep_dialog.agents import AgentDQN
from deep_dialog.usersims import RealUser
from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg
from deep_dialog.dialog_system import StateTracker

# Global variables for models
agent = None
user_sim = None
dialog_manager = None
state_tracker = None

def initialize_bot():
    global agent, user_sim, dialog_manager, state_tracker
    
    # Configuration des paramètres
    params = {
        'diaact_nl_pairs': './deep_dialog/data/dia_act_nl_pairs.v6.json',
        'max_turn': 20,
        'episodes': 1,
        'slot_err_prob': 0.05,
        'slot_err_mode': 0,
        'intent_err_prob': 0.05,
        'epsilon': 0,
        'nlg_model_path': './deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p',
        'nlu_model_path': './deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p',
        'act_level': 0,
        'run_mode': 0,
        'auto_suggest': 0,
        'cmd_input_mode': 0,
        'experience_replay_pool_size': 1000,
        'dqn_hidden_size': 60,
        'batch_size': 16,
        'gamma': 0.9,
        'predict_mode': False,
        'warm_start': 1,
        'warm_start_epochs': 100,
        'trained_model_path': './deep_dialog/checkpoints/agt_9_287_340_0.89200.p',
        'write_model_dir': './deep_dialog/checkpoints/',
        'save_check_point': 10,
        'success_rate_threshold': 0.3,
        'split_fold': 5,
        'learning_phase': 'all',
    }

    # Charger les données et les modèles
    movie_kb_path = './deep_dialog/data/movie_kb.1k.p'
    with open(movie_kb_path, "rb") as f:
        movie_kb = pickle.load(f, encoding="latin1")

    act_set = text_to_dict('./deep_dialog/data/dia_acts.txt')
    slot_set = text_to_dict('./deep_dialog/data/slot_set.txt')
    dict_path = './deep_dialog/data/dicts.v3.p'
    movie_dictionary = pickle.load(open(dict_path, 'rb'))

    agent_params = {
        'max_turn': params['max_turn'],
        'epsilon': params['epsilon'],
        'agent_run_mode': params['run_mode'],
        'agent_act_level': params['act_level'],
        'experience_replay_pool_size': params['experience_replay_pool_size'],
        'dqn_hidden_size': params['dqn_hidden_size'],
        'batch_size': params['batch_size'],
        'gamma': params['gamma'],
        'predict_mode': params['predict_mode'],
        'trained_model_path': params['trained_model_path'],
        'warm_start': params['warm_start'],
        'cmd_input_mode': params['cmd_input_mode']
    }

    agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
    user_sim = RealUser(movie_dictionary, act_set, slot_set)

    nlg_model = nlg()
    nlg_model.load_nlg_model(params['nlg_model_path'])
    nlg_model.load_predefine_act_nl_pairs(params['diaact_nl_pairs'])
    agent.set_nlg_model(nlg_model)
    user_sim.set_nlg_model(nlg_model)

    nlu_model = nlu()
    nlu_model.load_nlu_model(params['nlu_model_path'])
    agent.set_nlu_model(nlu_model)
    user_sim.set_nlu_model(nlu_model)

    dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, movie_kb)
    state_tracker = StateTracker(act_set, slot_set, movie_kb)
    state_tracker.initialize_episode()
    user_action = user_sim.initialize_episode()
    agent.initialize_episode()
    state_tracker.update(user_action=user_action)

def get_bot_response(user_input):
    global agent, user_sim, dialog_manager, state_tracker
    if not agent or not user_sim or not dialog_manager or not state_tracker:
        initialize_bot()

    user_action = {'diaact': 'inform', 'inform_slots': {'input': user_input}, 'request_slots': {}, 'turn': 0, 'nl': user_input}
    state_tracker.update(user_action=user_action)
    state = state_tracker.get_state_for_agent()
    agent_action = agent.state_to_action(state)
    state_tracker.update(agent_action=agent_action)
    agent.add_nl_to_action(agent_action)
    sys_action = state_tracker.dialog_history_dictionaries()[-1]
    user_action, episode_over = user_sim.next(sys_action)
    if episode_over:
        state_tracker.initialize_episode()
    return agent_action['act_slot_response']['nl']

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--simulation_epoch_size', type=int, default=50)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--success_rate_threshold', type=float, default=0.3)
    args = parser.parse_args()

    initialize_bot()
    while True:
        user_input = input("You: ")
        response = get_bot_response(user_input)
        print("Bot: ", response)
