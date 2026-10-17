1. **CARLA + TF++ setup**  
   Install CARLA/TF++, freeze one checkpoint, and successfully complete closed-loop routes.

2. **Instrument TF++ outputs**  
   Log path predictions, full target-speed distribution, controls, ego state, and route context every timestep.

3. **Set up 2–3 scenario families**  
   Select Bench2Drive/custom roundabout, lane split/merge, obstacle-detour scenarios and make runs repeatable.

4. **Generate closed-loop dataset**  
   Run many scenario variations and collect enough **failures + successful/benign corrections**.

5. **Create labels and dataset splits**  
   Label each timestep by whether TF++ fails within the next 3 s; create leakage-safe train/val/test splits.

6. **Build simple temporal features**  
   Compute path disagreement, target-speed distribution divergence, steering/control variation, etc. over time.

7. **Evaluate temporal signal itself**  
   Determine whether those simple temporal signals predict failure earlier than geometric hazards or TF++ confidence.

8. **Train GRU temporal monitor**  
   Feed 1–3 s histories into a small GRU and test whether learned temporal structure beats simple temporal features.

9. **Implement common safe-stop intervention**  
   When any monitor triggers, release throttle and apply the same bounded-braking fallback.

10. **Final paired closed-loop evaluation**  
    Compare no monitor vs geometry vs TF++ confidence vs temporal features vs GRU on failures prevented, warning time, false stops, and route completion.

11. **Ablations + final analysis**  
    Test history length/input groups and conclude whether temporal instability adds value and whether the GRU adds further value.