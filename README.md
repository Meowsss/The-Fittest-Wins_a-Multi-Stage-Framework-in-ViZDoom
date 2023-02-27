## Usage
`Python>=3.6` is required. We've tested `Python 3.6.5`.

### Minimal Working Example
To use the TLeague framework and run a minimal training,
one needs to install the following basic packages:
* [TLeague](vizdoom/VDAIC2017v2/MyPlayer/TLeague/README.md): the main logic of Competitive SelfPlay MultiAgent Reinforcement Learning.
* [TPolicies](vizdoom/VDAIC2017v2/MyPlayer/TPolicies/README.md): a lib for building Neural Net used in RL and IL.
* [Arena](vizdoom/VDAIC2017v2/MyPlayer/Arena/README.md): a lib of environments and env-agent interfaces.

See the docs therein for how to install `TLeague`, `TPolicies`, `Arena`, respectively.
Briefly, 
it amounts to git-cloning/downloading the repos and do the in-place pip installation. 
For examples,
```bash
cd ~/vizdoom/VDAIC2017v2/MyPlayer/TLeague && pip install -e . && cd ~
cd ~/vizdoom/VDAIC2017v2/MyPlayer/TPolicies && pip install -e . && cd ~
cd ~/vizdoom/VDAIC2017v2/MyPlayer/Arena && pip install -e . && cd ~
# manually install tensorflow 1.15.0 as required by TPolicies
pip install tensorflow==1.15.0
```
### ViZDoom Training
When installing the `Arena` package, 
one needs additionally install ViZDoom (>=1.1.8),

Refer also to the [link here](vizdoom/build_docker/README.md) for how to (auto-)build the docker image,
which is yet-another guide to installation from scratch.

For running training over a k8s cluster, see the [link here](vizdoom/README.md#training-code).
