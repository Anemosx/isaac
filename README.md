# Hidden Attacks in Multi-Agent Reinforcement Learning

![isaac_60](https://github.com/Anemosx/master-isaac/blob/main/isaac_60.gif)

## Abstract
Multi-Agent Reinforcement Learning has become more and more important in recent
years. Whether in the areas of logistics, robotics or transportation, everywhere the objective
is to increase the performance. While this has been the main focus, the question
of robustness and resilience of such agents has become increasingly relevant. Adversarial
Attacks can often affect agents, resulting in a decrease in performance or even total failure.
While many works around Adversarial Attacks have been focused on the perception
and/or the environment, the attack vector originating from agents has been less investigated.
This may involve agents being affected by safety issues, such as malfunctions
or outages. On the other hand, agents may also be manipulated by external malicious
intent. In this work we will take a closer look at the security aspect and the resulting
impact of such externally compromised agents (attackers) on other actors (protagonists).
In this process we introduce Infiltrating Stealth Agent Attack Controller (ISAAC), a new
approach which leverages this attack vector to reduce the performance of the protagonists
while at the same time remaining hidden with respect to external observers. For
this purpose we design our Adversarial Attack to be natural and a black box attack, thus
representing a scenario which is as close to real life as possible. Thereby we simulate a
malicious attack that does not possess any additional information or knowledge regarding
the protagonists. In order to evaluate ISAAC, we will look at various state-of-the-art
algorithms and perform attacks on them. Additionally, to measure the success of our
approach using a number of different metrics, we will utilize several scenarios from the
StarCraft Multi-Agent Challenge (SMAC). In the SMAC environments agents have to
cooperate with each other to be successful in a continuous setting. During this process
we will observe that for various algorithms ISAAC is able to significantly decrease the
performance of the protagonists while remaining hidden. As a result we provide a benchmark
tool to study and measure the robustness and resilience of algorithms in multi-agent
systems with respect to Adversarial Attacks originating from agents.

[Presentation](https://github.com/Anemosx/master-isaac/blob/main/master_isaac_pres.pdf)

[Full Thesis](https://github.com/Anemosx/master-isaac/blob/main/Masterthesis_ISAAC.pdf)

![trade_off_isaac](https://github.com/Anemosx/master-isaac/blob/main/isaac_trade_off.png?raw=true)

![position_isaac](https://github.com/Anemosx/master-isaac/blob/main/isaac_positioning.png?raw=true)


## Citation

If you find this work useful, please cite it as follows:

```bibtex
@misc{unterauer2022isaac,
      title={Hidden Attacks in Multi-Agent Reinforcement Learning}, 
      author={Arnold Unterauer},
      year={2022},
      eprint={},
      archivePrefix={},
      primaryClass={},
      url={https://github.com/Anemosx/isaac} 
}
