# ICV Shapley MARL

Implementation of the paper *Understanding Action Effects through Instrumental Empowerment in Multi-Agent Reinforcement Learning*, presented at ECAI 2025.

## Usage

The code was implemented using Python 3.10. In addition, install `requirements.txt` packages for the environments LBF and MPE. Check the modifications to the original MPE Tag environment to slightly adjust the observation, as described in `mpe_env/spread_tag_env.py`. For these two environments, we use pre-trained models through Ray RLlib. 

For GFootball, you may need additional packages to install, proceed as described in [their repository](https://github.com/google-research/football). We use the pre-trained model [TiZero](https://github.com/OpenRL-Lab/TiZero), which is directly included in this repository.

For all environments, there is a dedicated `test_*.py` file to first inspect the environment.

### LBF and MPE

The files `policy_distribution.py` plot the action probabilities and value function while allowing for manual action input. `policy_history.py` records the absolute values and differences between two consecutive time steps, but without player contributions (i.e. no ICV yet). The file `policy_icv.py` is the one to measure the *Intended Cooperation Values* (ICV) and store the histories and bar plots. All outputs will be stored in `results_store`. 

### GFootball

For GFootball, we consider two cases: `grf_icv_ballplayer.py` computes the action effects either on the ball player or on the striker, whereas `grf_icv_forward_trio.py` measures the effects on the forwarding roles (right-midfield, center-forward, and left-midfield). Results are stored in `icv_store_*`. 

## Citation

```bibtex
@inproceedings{icv_ecai25,
  title     = {Understanding Action Effects through Instrumental Empowerment in Multi-Agent Reinforcement Learning},
  author    = {Ardian, Selmonaj and Miroslav, Strupl and Oleg, Szehr and Alessandro, Antonucci},
  booktitle = {Proceedings of the European Conference on Artificial Intelligence, {ECAI}},
  year      = {2025}
}
```