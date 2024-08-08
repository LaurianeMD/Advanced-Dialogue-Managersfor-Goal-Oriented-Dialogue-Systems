import gradio as gr
import pickle
from deep_dialog.dialog_system import DialogManager, text_to_dict, StateTracker
from deep_dialog.agents import AgentDQN
from deep_dialog.usersims import RealUser
from deep_dialog.nlu import nlu
from deep_dialog.nlg import nlg

# Charger les modèles et les données
movie_kb_path = './deep_dialog/data/movie_kb.1k.p'
with open(movie_kb_path, "rb") as f:
    movie_kb = pickle.load(f, encoding="latin1")

act_set = text_to_dict('./deep_dialog/data/dia_acts.txt')
slot_set = text_to_dict('./deep_dialog/data/slot_set.txt')
dict_path = './deep_dialog/data/dicts.v3.p'
with open(dict_path, 'rb') as f:
    movie_dictionary = pickle.load(f)

# Paramètres de l'agent
agent_params = {
    'max_turn': 20,
    'epsilon': 0,
    'agent_run_mode': 0,
    'agent_act_level': 0,
    'experience_replay_pool_size': 1000,
    'dqn_hidden_size': 60,
    'batch_size': 16,
    'gamma': 0.9,
    'predict_mode': False,
    'trained_model_path': './deep_dialog/checkpoints/agt_9_287_340_0.89200.p',
    'warm_start': 1,
    'cmd_input_mode': 0
}

agent = AgentDQN(movie_kb, act_set, slot_set, agent_params)
user_sim = RealUser(movie_dictionary, act_set, slot_set)

# Charger les modèles NLG et NLU
nlg_model = nlg()
nlg_model.load_nlg_model('./deep_dialog/models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p')
nlg_model.load_predefine_act_nl_pairs('./deep_dialog/data/dia_act_nl_pairs.v6.json')
agent.set_nlg_model(nlg_model)
user_sim.set_nlg_model(nlg_model)

nlu_model = nlu()
nlu_model.load_nlu_model('./deep_dialog/models/nlu/lstm_[1468447442.91]_39_80_0.921.p')
agent.set_nlu_model(nlu_model)
user_sim.set_nlu_model(nlu_model)

# Initialiser le gestionnaire de dialogue
dialog_manager = DialogManager(agent, user_sim, act_set, slot_set, movie_kb)
state_tracker = StateTracker(act_set, slot_set, movie_kb)
state_tracker.initialize_episode()

def user_query(user_input):
    try:
        # Utiliser le modèle NLU pour analyser l'entrée utilisateur
        user_action = nlu_model.generate_dia_act(user_input)  # Utiliser generate_dia_act pour NLU
        state_tracker.update(user_action=user_action)
        
        state = state_tracker.get_state_for_agent()
        agent_action = agent.state_to_action(state)
        state_tracker.update(agent_action=agent_action)
        agent.add_nl_to_action(agent_action)
        
        response = agent_action['act_slot_response']['nl']
        return response
    except Exception as e:
        return f"Erreur: {str(e)}"

# Définir l'interface Gradio
iface = gr.Interface(
    fn=user_query,
    inputs=gr.Textbox(lines=2, placeholder="Entrez votre requête ici..."),
    outputs="text",
    title="Dialogue System Interaction",
    description="Entrez une requête pour interagir avec le système de dialogue."
)

if __name__ == "__main__":
    iface.launch(share=True)
