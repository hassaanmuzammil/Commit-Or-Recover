# Commit or Recover

## A Temporal Safety Monitor for TransFuser++

**Course:** Intelligent Autonomous Systems  
**Team Members:** Daniel Cai, Hassaan Muzammil, Nhi Nguyen  
**Selected Track:** Safety and Evaluation  
**Secondary Area:** End-to-End Driving

## Abstract

End-to-end autonomous-driving systems can produce unsafe behavior when their intended motion becomes unstable near difficult decision points such as roundabouts, lane splits, merges, and obstacle detours. A policy may alternate between incompatible paths or issue rapidly changing control commands before a conventional collision detector recognizes an immediate hazard. This project investigates whether those temporal changes can provide an early warning of failure.

We will use a pretrained **TransFuser++ (TF++)** agent as a frozen base driving policy in the CARLA simulator. Rather than modifying or retraining TF++, we will develop an independent temporal safety monitor that observes a short history of the policy's predicted paths, target-speed probability distributions, steering, throttle, brake, ego state, and navigation information. A lightweight gated recurrent unit (GRU) will estimate the probability that the unchanged policy will cause a collision, off-road violation, or route-deviation failure within a fixed prediction horizon. When the predicted risk exceeds a calibrated threshold, the monitor will invoke a fixed safe-stop controller.

The proposed method will be evaluated in closed loop against four conditions: unmonitored TF++, an instantaneous geometric safety check, a confidence threshold based on TF++'s target-speed distribution, and a handcrafted temporal-instability detector. All monitoring methods will use the same driving policy, scenarios, fallback behavior, and matched intervention budget. The primary goal is to determine whether temporal policy instability prevents more failures than reactive safety checks without causing an unacceptable number of unnecessary stops or reducing route completion.

## Research Question

> Does temporal inconsistency in TransFuser++'s predicted paths, target-speed distributions, and control outputs provide earlier and more selective warning of closed-loop driving failures than instantaneous geometric safety checks or single-frame confidence measures?


## Disclaimer

This is an academic simulation project. The proposed monitor and fallback controller are not intended for deployment on public roads or use in a production vehicle.
