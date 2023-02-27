# ViZDoom Experiments
This page contains the resources for the experiments of ViZDoom as discussed in the TLeague technical report.

## Training Code
There are three stages for training:
one for the navigation, and the other for the frag.
We provide the corresponding `.yml.jinja2` files here: [for navigation](vdtr-navi-open.yml.jinja2), [for frag](vdtr-frag-open.yml.jinja2) and [for adaptive](vdtr-adaptive-open.yml.jinja2) respectively.

Run the training over a k8s cluster:
```bash
# start
python render_template.py vdtr-navi-open.yml.jinja2 | kubectl apply -f -
# stop
python render_template.py vdtr-navi-open.yml.jinja2 | kubectl delete -f -
```
```bash
# start
python render_template.py vdtr-frag-open.yml.jinja2 | kubectl apply -f -
# stop
python render_template.py vdtr-frag-open.yml.jinja2 | kubectl delete -f -
```
```bash
# start
python render_template.py vdtr-frag-open.yml.jinja2 | kubectl apply -f -
# stop
python render_template.py vdtr-frag-open.yml.jinja2 | kubectl delete -f -
```

## Evaluation
[The evaluation code](VDAIC2017v2/README.md) is a modification over the original evaluation code https://github.com/mihahauke/VDAIC2017

```bash
bash MyPlayer/TLeague/tleague/sandbox/example_evaluation_vd.sh evaluation
```

