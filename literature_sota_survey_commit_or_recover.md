# Literature and State-of-the-Art Survey

## Commit or Recover: A Temporal Safety Monitor for Detecting Policy Instability in End-to-End Driving

**Primary track:** Safety and Evaluation  
**Secondary area:** End-to-End Driving  
**Survey window:** 2023-2026  
**Project focus:** Detecting whether a frozen TransFuser++ policy is becoming unstable from the recent history of its predicted spatial paths, target-speed distributions, controls, ego state, and route information, then invoking a fixed safe fallback before a collision, off-road event, or route-deviation failure.

## 1. Scope of the Survey

The proposed project lies at the intersection of four areas:

1. TransFuser++ and its path-plus-target-speed output representation;
2. runtime monitors that predict imminent safety failures;
3. instantaneous uncertainty versus temporal output inconsistency;
4. recovery systems that intervene after a potential failure is detected.

The ten works below were selected because they contribute the chosen base policy, a closely competing monitoring method, a relevant failure-learning strategy, a closed-loop evaluation benchmark, or a recovery mechanism. The closest works are **EvoEye**, **Argus**, **System Safety Monitoring using Temporal Metric Forecasting**, and **ADReFT**. Their overlap means this project should not claim to be the first learned runtime monitor, temporal failure predictor, or online recovery system. Its narrower contribution is a controlled evaluation of whether **temporal inconsistency in a frozen TransFuser++ policy's observable path, target-speed, and control outputs** provides warning beyond instantaneous geometric risk and TransFuser++'s own single-frame confidence.

## 2. State of the Art at a Glance

| # | Year | Paper / model | Main contribution | Relationship to Commit or Recover |
|---:|:---:|---|---|---|
| 1 | 2026 | [EvoEye](https://arxiv.org/abs/2607.03755) | Learns temporal interactions among observable ADS runtime signals and evolves the monitor through targeted scenario acquisition | Closest recent methodological competitor; increases the need to isolate TF++-specific temporal signals |
| 2 | 2023 | [Hidden Biases / TransFuser++](https://arxiv.org/abs/2306.07957) | Introduces the chosen policy, with spatial path prediction and target-speed classification | Frozen policy under test and source of path, speed-confidence, and control histories |
| 3 | 2023/24 | [MarMot](https://arxiv.org/abs/2310.07414) | Runtime monitoring through metamorphic input-output consistency | Strong black-box monitoring baseline, but not based on temporal action instability |
| 4 | 2024 | [Predicting Safety Misbehaviours using UQ](https://arxiv.org/abs/2404.18573) | Uses MC Dropout and Deep Ensembles for early failure warning | Provides uncertainty baselines and early-warning evaluation methodology |
| 5 | 2024 | [System Safety Monitoring using Temporal Metric Forecasting](https://arxiv.org/abs/2405.13254) | Forecasts future safety metrics from learned-component outputs and context | Closest prior work on temporal safety monitoring; motivates careful differentiation |
| 6 | 2024 | [Validity Learning on Failures](https://arxiv.org/abs/2406.01544) | Collects closed-loop failure states and learns which candidate trajectories are valid | Supports failure-data collection, but repairs the planner offline instead of monitoring it online |
| 7 | 2024 | [Bench2Drive](https://arxiv.org/abs/2406.03877) | Multi-ability, closed-loop CARLA benchmark with short interactive scenarios | Provides scenario design, metrics, and generalization tests |
| 8 | 2025/26 | [EnDfuser](https://arxiv.org/abs/2506.00560) | Diffusion planner whose trajectory samples expose planning uncertainty | Conceptual comparison between within-frame sample diversity and cross-time TF++ inconsistency |
| 9 | 2025 | [ADReFT](https://arxiv.org/abs/2506.23960) | Joint state monitor and adaptive online decision repair | Most direct competitor; establishes the main novelty risk |
| 10 | 2025 | [Argus](https://arxiv.org/abs/2511.09032) | Monitors end-to-end ADS trajectories and invokes an IDM-based hazard mitigator | Closest monitor-and-takeover framework; motivates separating instability detection from fallback quality |

## 3. Individual Paper and Model Summaries

### 1. EvoEye: Self-Evolving Runtime Monitoring for Autonomous Driving Systems

**Citation:** M. Cheng, L. Briand, and X. Xie. "EvoEye: Self-Evolving Runtime Monitoring for Autonomous Driving Systems." arXiv:2607.03755, 2026. [Paper](https://arxiv.org/abs/2607.03755)

**Problem and approach.** EvoEye addresses two limitations of runtime monitors: their dependence on a fixed training set and their inability to represent risks that emerge from interactions across an autonomous-driving stack. Its FusionMonitor encodes observable perception, prediction, planning, and control messages over a temporal window to predict an imminent collision. BlindSpotEvolver then uses monitor errors to guide scenario mutation and acquire additional training executions.

**Main result.** In CARLA experiments with Baidu Apollo, the authors report that FusionMonitor improved recall at a fixed false-positive rate while maintaining millisecond-scale latency and several seconds of median warning time. Monitor-guided scenario acquisition also outperformed uniform and violation-oriented sampling under a common simulation budget.

**Relevance to this project.** EvoEye is the closest current methodological comparison because it also learns from temporal runtime signals and labels windows preceding collisions. It reinforces the importance of fixed-false-positive-rate evaluation, warning time, runtime overhead, and hard negative examples.

**Difference and novelty risk.** EvoEye monitors heterogeneous messages from a modular ADS and adds a self-evolving data-acquisition loop. Commit or Recover instead isolates a smaller question in a frozen end-to-end policy: whether TF++ path, target-speed, and control inconsistency predicts decision-point failures beyond current-frame confidence and geometry. The project should not claim temporal runtime prediction as new; its contribution is the TF++-specific signal study and controlled closed-loop comparison.

---

### 2. Hidden Biases of End-to-End Driving Models / TransFuser++

**Citation:** B. Jaeger, K. Chitta, and A. Geiger. "Hidden Biases of End-to-End Driving Models." *IEEE/CVF International Conference on Computer Vision (ICCV)*, 2023. [Paper](https://arxiv.org/abs/2306.07957) | [Code and models](https://github.com/autonomousvision/carla_garage)

**Problem and approach.** Jaeger et al. investigate why many CARLA end-to-end systems improve despite simultaneous changes to models, data, controllers, and other implementation details. They identify two recurring biases: target-point conditioning can act as a shortcut that helps lateral recovery but may cause catastrophic steering when the target is far away, and deterministic waypoint regression can average over multimodal future behaviors.

The authors introduce TransFuser++ and test alternatives including a stronger decoder, path-based outputs, augmentation for lateral recovery, and target-speed classification to represent longitudinal uncertainty more explicitly.

**Main result.** Their controlled experiments show that output representation and target-point conditioning materially affect collision avoidance and recovery. In particular, deterministic waypoint outputs can interpolate between possible behaviors instead of selecting a coherent mode.

**Relevance to this project.** TransFuser++ is the chosen frozen policy under test. Unlike original TransFuser, TF++ disentangles lateral and longitudinal behavior: it predicts a spatial path sampled by distance and a probability distribution over target-speed classes. The monitor can therefore compare successive ego-aligned paths, changes in the speed distribution, controller outputs, ego motion, and target-point alignment. Target-speed entropy or confidence margin also provides an unusually strong single-frame uncertainty baseline.

**Difference and limitation.** TransFuser++ improves the driving policy itself and already uses confidence-weighted target speed, but it does not provide an external temporal monitor that predicts an impending failure or decides when to invoke a fallback. Commit or Recover keeps TF++ frozen and tests whether residual inconsistency across its output history adds value beyond the uncertainty exposed at one frame.

---

### 3. Metamorphic Runtime Monitoring of Autonomous Driving Systems (MarMot)

**Citation:** J. Ayerdi, A. Iriarte, P. Valle, I. Roman, M. Illarramendi, and A. Arrieta. "Metamorphic Runtime Monitoring of Autonomous Driving Systems." arXiv:2310.07414, first released 2023 and revised 2024. [Paper](https://arxiv.org/abs/2310.07414)

**Problem and approach.** MarMot addresses runtime confidence estimation when an end-to-end system may encounter external anomalies, such as fog, or internal faults, such as errors caused by mislabeled training data. It applies domain-specific metamorphic transformations to an input and examines whether the policy's outputs obey expected consistency relations. The resulting consistency score functions as an uncertainty signal.

**Main result.** The study evaluates five metamorphic relations on a small-scale physical vehicle and a simulated system. The paper reports detection of up to 65% of external and 100% of internal anomalies for the physical system, and up to 54% and 88%, respectively, for the simulated system. It was competitive with or better than SelfOracle, MC Dropout, and ensemble-based monitors in the reported settings.

**Relevance to this project.** MarMot is an important black-box baseline because it estimates policy reliability without relying on the model's hidden activations. It also reinforces that consistency, not only raw confidence, can be a useful runtime signal.

**Difference and limitation.** MarMot measures consistency across transformed versions of the current input. Commit or Recover measures consistency across TF++'s naturally occurring path, target-speed, and control sequence. The proposed temporal monitor therefore needs only one policy execution per timestep, whereas metamorphic monitoring may require multiple transformed evaluations.

---

### 4. Predicting Safety Misbehaviours in Autonomous Driving Systems using Uncertainty Quantification

**Citation:** R. Grewal, P. Tonella, and A. Stocco. "Predicting Safety Misbehaviours in Autonomous Driving Systems using Uncertainty Quantification." IEEE conference publication, 2024. [Paper](https://arxiv.org/abs/2404.18573) | [IEEE record](https://ieeexplore.ieee.org/document/10638606)

**Problem and approach.** This study tests whether Bayesian uncertainty measures can predict an end-to-end driving failure before it occurs. It applies MC Dropout and Deep Ensembles to DAVE-2-style lane-keeping models and raises an alert when uncertainty exceeds a threshold learned from nominal driving.

**Main result.** The authors evaluate more than 250 failures across out-of-distribution weather and mutated faulty models in the Udacity simulator. They report that the uncertainty methods frequently warn several seconds before an out-of-bounds event. Their best configuration achieved a reported 94% F3 score and improved failure detection over SelfOracle and ThirdEye while considering runtime overhead.

**Relevance to this project.** This paper provides useful evaluation concepts: detection windows, early-warning time, false alarms, and inference overhead. For the minimum viable experiment, TF++'s target-speed entropy and confidence margin provide a cheaper, policy-native uncertainty baseline. MC Dropout or a multi-checkpoint ensemble can remain a stretch comparison if compute permits.

**Difference and limitation.** The study uses white-box access to relatively small lane-keeping networks and focuses largely on weather or model-mutation failures. Commit or Recover targets complex CARLA decision points and an output-level monitor operating on TF++ path, speed-distribution, and control history. It also evaluates an actual closed-loop fallback rather than warning quality alone.

---

### 5. System Safety Monitoring of Learned Components Using Temporal Metric Forecasting

**Citation:** S. Sharifi, A. Stocco, and L. C. Briand. "System Safety Monitoring of Learned Components Using Temporal Metric Forecasting." arXiv:2405.13254, 2024. [Paper](https://arxiv.org/abs/2405.13254)

**Problem and approach.** The authors propose forecasting a quantitative safety metric from a learned component's outputs and the current operational context. Rather than detecting only a present anomaly, a probabilistic time-series model predicts whether a system-level safety requirement will be violated in the future. Four forecasting architectures are evaluated in autonomous aviation and autonomous-driving case studies.

**Main result.** The study finds that temporal forecasting can predict future safety-metric violations with practical latency and resource usage. Among the tested models, the Temporal Fusion Transformer produced the most accurate imminent-violation predictions in both case studies.

**Relevance to this project.** This is the closest methodological predecessor to the proposed GRU monitor. It strongly supports using rolling component-output histories and reporting both prediction quality and runtime cost. It should influence the baseline set: a simple GRU can be the minimum viable model, while a Temporal Fusion Transformer can be a stretch comparison.

**Difference and novelty risk.** Both projects forecast failure from temporal component behavior. Commit or Recover must therefore be framed more narrowly around **TF++ output instability at discrete driving decisions**, including path-mode switching, target-speed distribution changes, steering reversals, and route-progress inconsistency. Its controlled comparison against current-frame geometry, TF++ confidence, and handcrafted oscillation at a matched intervention rate is the clearest differentiator.

---

### 6. Validity Learning on Failures: Mitigating the Distribution Shift in Autonomous Vehicle Planning

**Citation:** F. Arasteh et al. "Validity Learning on Failures: Mitigating the Distribution Shift in Autonomous Vehicle Planning." 2024. [Paper](https://arxiv.org/abs/2406.01544) | [IEEE record](https://ieeexplore.ieee.org/document/11128610)

**Problem and approach.** Imitation-learned planners often encounter states that are absent from expert demonstrations. The authors roll out a pretrained planner, collect states where it violates objectives such as obstacle clearance or traffic rules, and build a failure dataset. Because these states do not have expert trajectories, they introduce a weakly supervised Validity Learning objective that increases the probability of candidate trajectories satisfying known safety and traffic constraints.

**Main result.** Experiments in reactive CARLA simulation and non-reactive log replay show improved closed-loop score, progress, and success rate. The work also studies the consistency between reactive and replay evaluation and compares with Bench2Drive baselines.

**Relevance to this project.** The data-generation strategy is directly useful: roll out a fixed policy, label windows preceding safety violations, and preserve hard negative cases. It also demonstrates that failure data can be collected without asking a human to provide a correct trajectory for every mistake.

**Difference and limitation.** Validity Learning modifies the planner offline so that it scores valid trajectories more highly. Commit or Recover does not retrain the planner; it performs runtime monitoring and invokes a fixed fallback. This distinction lets the proposed experiment isolate detection from policy repair.

---

### 7. Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving

**Citation:** X. Jia, Z. Yang, Q. Li, Z. Zhang, and J. Yan. "Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving." *NeurIPS Datasets and Benchmarks Track*, 2024. [Paper](https://arxiv.org/abs/2406.03877) | [Official repository](https://github.com/Thinklab-SJTU/Bench2Drive)

**Problem and approach.** Bench2Drive responds to limitations of open-loop trajectory error and long, high-variance CARLA routes. It supplies a common training dataset and evaluates end-to-end policies on short, interactive closed-loop scenarios intended to isolate different driving abilities.

**Main result.** The release contains approximately two million annotated frames from 13,638 clips and evaluates 44 interactive scenario types across 23 weather conditions and 12 towns. Its evaluation suite totals 220 routes and reports scenario-specific driving capabilities and infractions.

**Relevance to this project.** Bench2Drive is the most relevant modern evaluation resource. Its short scenarios, including detours, cut-ins, and obstacle interactions, are better suited to repeated matched-condition experiments than a single long route. Its infraction and route-completion metrics can be combined with monitor-specific measures such as warning time and unnecessary intervention rate.

**Difference and scope decision.** Bench2Drive is a broad benchmark, not a failure monitor. The full two-million-frame training set is unnecessary for the minimum viable project because the base policy remains frozen. The team can reuse selected scenario definitions and evaluation ideas while collecting only the temporal logs required for monitor training.

---

### 8. Using Ensemble Diffusion to Estimate Uncertainty for End-to-End Autonomous Driving (EnDfuser)

**Citation:** F. Wintel, S. H. Hoeg, G. Kiss, and F. Lindseth. "Using Ensemble Diffusion to Estimate Uncertainty for End-to-End Autonomous Driving." arXiv:2506.00560, 2025; accepted at NLDL 2026. [Paper](https://arxiv.org/abs/2506.00560)

**Problem and approach.** EnDfuser replaces a point-estimate trajectory head with a diffusion-based planner. From a single fused camera-LiDAR observation, it samples a distribution of candidate trajectories; the reported implementation generates 128 samples. Dispersion and multimodality in these samples provide an interpretable estimate of planning uncertainty.

**Main result.** The authors use the sampled trajectory distribution to define a simple safety rule and report a 1.7% improvement in driving score on the LAV benchmark. The work shows how trajectory-level uncertainty can be obtained from the planner's outputs instead of a separate perception uncertainty model.

**Relevance to this project.** EnDfuser suggests a powerful baseline: measure disagreement among candidate plans at one timestep. This can be compared with disagreement across successive plans, which is the proposed project's temporal signal.

**Difference and limitation.** EnDfuser requires a specialized diffusion policy and evaluates uncertainty within a single observation. Commit or Recover uses temporal history from a fixed deterministic TF++ path head together with its discrete target-speed distribution. EnDfuser is therefore a conceptual comparison rather than a required implementation baseline.

---

### 9. ADReFT: Adaptive Decision Repair for Safe Autonomous Driving via Reinforcement Fine-Tuning

**Citation:** M. Cheng, X. Xie, R. Wang, Y. Zhou, and M. Hu. "ADReFT: Adaptive Decision Repair for Safe Autonomous Driving via Reinforcement Fine-Tuning." arXiv:2506.23960, 2025. [Paper](https://arxiv.org/abs/2506.23960)

**Problem and approach.** ADReFT argues that existing online repair systems can be inflexible or excessively conservative. It introduces a Transformer with two jointly trained heads: a **State Monitor** estimates the severity of the current safety-critical state, and a **Decision Adapter** outputs an adaptive repair action. The model is first pretrained using coarse labels that mark states preceding violations and then reinforcement-fine-tuned to improve the safety-versus-driving-quality tradeoff.

**Main result.** The reported evaluation shows improved online repair performance over the paper's comparison methods, with the reinforcement stage intended to reduce the conservatism of coarse supervised repair.

**Relevance to this project.** ADReFT is the closest prior work and should be discussed prominently in the proposal. It validates the general premise that future violations can label precursor states and that intervention quality must be evaluated together with safety.

**Difference and novelty risk.** A broad claim such as "we propose a learned monitor that detects unsafe driving and repairs the decision" would overlap heavily with ADReFT. The project should instead make the following controlled claim:

> We test whether temporal inconsistency in a frozen TransFuser++ policy's path, target-speed, and control history predicts decision-point failures beyond current-frame geometric monitoring and target-speed confidence, while holding the fallback action and intervention budget fixed.

Unlike ADReFT, the proposed minimum viable experiment intentionally does not learn the repair action. That reduction in scope becomes a scientific advantage because it isolates whether the temporal instability signal itself is useful.

---

### 10. Argus: Resilience-Oriented Safety Assurance Framework for End-to-End ADSs

**Citation:** D. Wang, Y. Lu, B. Chen, S. Hao, H. Jiang, Y. Tian, and X. Peng. "Argus: Resilience-Oriented Safety Assurance Framework for End-to-End ADSs." arXiv:2511.09032, 2025. [Paper](https://arxiv.org/abs/2511.09032)

**Problem and approach.** Argus is a modular runtime-assurance framework for end-to-end driving systems. It monitors planned trajectories for collision, stop-signal, and stalling hazards, maintains takeover and recovery buffers, and transfers control to an IDM-based hazard mitigator that can adjust speed or reroute around obstacles.

**Main result.** The authors integrate Argus with TCP, UniAD, and VAD and evaluate it on Bench2Drive and CARLA Leaderboard 2.0. They report prevention of a substantial portion of safety violations with low monitoring overhead, demonstrating that an external monitor-and-takeover layer can improve several end-to-end policies.

**Relevance to this project.** Argus is the closest systems-level competitor. It supports the use of a frozen policy, an external monitor, temporal buffering, closed-loop takeover evaluation, and metrics that jointly measure safety and continued driving performance.

**Difference and novelty risk.** Argus mainly determines whether the current planned motion will create a physical hazard and uses a comparatively capable IDM-based mitigator. Commit or Recover asks whether instability in TF++'s own outputs warns of failure before a current-path geometric check activates. Using the same fixed safe stop for every monitor isolates detection quality from replanning quality.



## 4. References

1. Cheng, M., Briand, L., and Xie, X. "EvoEye: Self-Evolving Runtime Monitoring for Autonomous Driving Systems." arXiv:2607.03755, 2026. <https://arxiv.org/abs/2607.03755>
2. Jaeger, B., Chitta, K., and Geiger, A. "Hidden Biases of End-to-End Driving Models." *ICCV*, 2023. <https://arxiv.org/abs/2306.07957>
3. Ayerdi, J., Iriarte, A., Valle, P., Roman, I., Illarramendi, M., and Arrieta, A. "Metamorphic Runtime Monitoring of Autonomous Driving Systems." arXiv:2310.07414, 2023/2024. <https://arxiv.org/abs/2310.07414>
4. Grewal, R., Tonella, P., and Stocco, A. "Predicting Safety Misbehaviours in Autonomous Driving Systems using Uncertainty Quantification." IEEE, 2024. <https://arxiv.org/abs/2404.18573>
5. Sharifi, S., Stocco, A., and Briand, L. C. "System Safety Monitoring of Learned Components Using Temporal Metric Forecasting." arXiv:2405.13254, 2024. <https://arxiv.org/abs/2405.13254>
6. Arasteh, F. et al. "Validity Learning on Failures: Mitigating the Distribution Shift in Autonomous Vehicle Planning." 2024. <https://arxiv.org/abs/2406.01544>
7. Jia, X., Yang, Z., Li, Q., Zhang, Z., and Yan, J. "Bench2Drive: Towards Multi-Ability Benchmarking of Closed-Loop End-to-End Autonomous Driving." *NeurIPS Datasets and Benchmarks*, 2024. <https://arxiv.org/abs/2406.03877>
8. Wintel, F., Hoeg, S. H., Kiss, G., and Lindseth, F. "Using Ensemble Diffusion to Estimate Uncertainty for End-to-End Autonomous Driving." arXiv:2506.00560, 2025; NLDL 2026. <https://arxiv.org/abs/2506.00560>
9. Cheng, M., Xie, X., Wang, R., Zhou, Y., and Hu, M. "ADReFT: Adaptive Decision Repair for Safe Autonomous Driving via Reinforcement Fine-Tuning." arXiv:2506.23960, 2025. <https://arxiv.org/abs/2506.23960>
10. Wang, D., Lu, Y., Chen, B., Hao, S., Jiang, H., Tian, Y., and Peng, X. "Argus: Resilience-Oriented Safety Assurance Framework for End-to-End ADSs." arXiv:2511.09032, 2025. <https://arxiv.org/abs/2511.09032>
