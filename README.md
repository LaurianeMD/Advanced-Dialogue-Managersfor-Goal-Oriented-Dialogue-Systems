# Advanced-Dialogue-Managers-for-Goal-Oriented-Dialogue-Systems
### README

Based off of the code repo [DenebUniverse](https://github.com/DenebUniverse/UserSimulator_Chatbot) and [paper End-to-End Task-Completion Neural Dialogue Systems](https://arxiv.org/pdf/1806.00780). This repo is a simplified version of [Master Thesis GO Chatbots](https://github.com/IlievskiV/Master_Thesis_GO_Chatbots)

# Deep Dialog System

This repository contains a comprehensive implementation of a Deep Dialogue System using Reinforcement Learning (RL). It includes various components such as agents, user simulators, and dialogue management to facilitate interaction in a conversational environment.

## Overview

The project is designed to build and evaluate dialogue management systems using different types of agents and user simulators. It integrates Natural Language Understanding (NLU) and Natural Language Generation (NLG) models to handle and generate dialogue responses. The core of the system is based on a reinforcement learning framework, which is used to train and optimize dialogue agents.

## Project structure
``` sh
.
├── deep_dialog
│   ├── agents               # Agent implementations (e.g., DQN)
│   ├── data                 # Data files (e.g., movie KB, dialog acts)
│   ├── dialog_system        # Core dialog management modules
│   ├── models               # Pre-trained models (NLG, NLU)
│   ├── nlg                  # Natural Language Generation modules
│   ├── nlu                  # Natural Language Understanding modules
│   └── usersims             # User simulators
├── checkpoints              # Directory for saving trained models
├── scripts                  # Utility scripts
├── chatbot_project          # Django project settings for web interface
├── README.md                # Project README file
└── requirements.txt         # Python dependencies
``` 
## Components

### 1. **Agents**
- **InformAgent**: Provides informative responses to the user's queries.
- **RequestAllAgent**: Requests all required information from the user.
- **RandomAgent**: Chooses actions randomly.
- **EchoAgent**: Repeats the user's input.
- **RequestBasicsAgent**: Requests basic information from the user.
- **AgentDQN**: A Deep Q-Network based agent for reinforcement learning.

### 2. **User Simulators**
- **RealUser**: Simulates real user interactions.
- **RuleSimulator**: Follows predefined rules for user interactions.

### 3. **NLU and NLG**
- **NLU**: Handles Natural Language Understanding.
- **NLG**: Handles Natural Language Generation.

### 4. **Dialogue Management**
- **DialogManager**: Manages the dialogue flow between the agent and the user.

## Installation

1. **Clone the Repository**
   ```sh
   git clone https://github.com/LaurianeMD/Advanced-Dialogue-Managersfor-Goal-Oriented-Dialogue-Systems
   cd deep-dialog
   ```

2. **Set Up Virtual Environment**
   ```sh
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. **Install Dependencies**
   ```sh
   pip install -r requirements.txt
   ```

## Configuration

You need to specify various paths and parameters for the dialogue system. These can be set via command-line arguments:

- `--dict_path`: Path to the .json dictionary file.
- `--movie_kb_path`: Path to the movie knowledge base file.
- `--act_set`: Path to the dialogue act set file.
- `--slot_set`: Path to the slot set file.
- `--goal_file_path`: Path to the user goals file.
- `--diaact_nl_pairs`: Path to the pre-defined dialogue act and NL pairs file.
- `--nlg_model_path`: Path to the NLG model file.
- `--nlu_model_path`: Path to the NLU model file.
- `--trained_model_path`: Path to the trained model.

## Running the System

To run the dialogue system, execute the main script with the desired parameters. For example:

```sh
python run_dialog.py --dict_path './data/dicts.v3.p' --movie_kb_path './data/movie_kb.1k.p' --act_set './data/dia_acts.txt' --slot_set './data/slot_set.txt' --goal_file_path './data/user_goals_first_turn_template.part.movie.v1.p' --diaact_nl_pairs './data/dia_act_nl_pairs.v6.json' --nlg_model_path './models/nlg/lstm_tanh_relu_[1468202263.38]_2_0.610.p' --nlu_model_path './models/nlu/lstm_[1468447442.91]_39_80_0.921.p'
```

## Commands for Django Integration

To run administrative tasks with Django, use:

```sh
python manage.py runserver
```

This assumes you have set up Django with your desired settings.

## Usage in Production

For production use, ensure you have trained models and pre-defined dialogue act and NL pairs files. Adjust paths and parameters according to your deployment environment.

## Contributing

Contributions are welcome! Please fork the repository, make your changes, and submit a pull request. Ensure that you follow the coding standards and include relevant tests.

## License

This project is licensed under the MIT License. 

## Collaborators

- **Oumar KRA**
- **Séna Kodjovi KUDAWOO**
- **Carel MOUSSE**
- **Paule Marcelle ELANGA**
- **Lauriane MBAGDJE DORENAN**
- **Amath Sow**
