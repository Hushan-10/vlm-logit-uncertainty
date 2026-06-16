# Experimental Results

## Experiment: Blur vs Uncertainty

We evaluate uncertainty on the **same prompt** using three images:

1. **Clear image**
2. **Slightly blurred image**
3. **Highly blurred image**

Hypothesis:

- Increasing blur reduces visual evidence → model uncertainty increases.

**_Experiment 1 (Clear Image)_**

![pexels-pixabay-53435](https://github.com/user-attachments/assets/b53f4818-3c36-4690-a323-21b96504d162)

**_Results_**

<img width="397" height="139" alt="Screenshot 2026-01-03 132535" src="https://github.com/user-attachments/assets/b6c9d0bd-d75d-464d-84ce-987b12da5fce" />

**_Experiment 2 (Slightly blurred Image)_**
![slighlty blur image](https://github.com/user-attachments/assets/6fd5249f-d6d4-4eb5-8b21-4c513fa0c824)

**_Results_**

<img width="446" height="145" alt="Screenshot 2026-01-03 133331" src="https://github.com/user-attachments/assets/a3918a4f-c0fc-4174-878e-62f2e097b4e0" />

**_Experiment 3 (Blurred Image)_**
![blurred image](https://github.com/user-attachments/assets/d3788b22-1901-40ae-a57e-3d532a36a9ff)

**_Results_**

<img width="455" height="145" alt="Screenshot 2026-01-03 133653" src="https://github.com/user-attachments/assets/46288973-32ac-4eac-b412-ed4af88b7862" />

| Condition   | Predicted token | Top1 prob | Top2 prob | Margin (Top1-Top2) | Entropy |
| ----------- | --------------- | --------: | --------: | -----------------: | ------: |
| Clear       | Tree            |    0.9694 |    0.0180 |             0.9514 |   0.198 |
| Slight blur | Tree            |    0.9071 |    0.0481 |              0.859 |   0.493 |
| Heavy blur  | Tree            |    0.8041 |    0.0821 |              0.722 |   0.822 |

## Key Decisions from the Blur Experiment (Logit-Based Uncertainty)

- **Blur reliably increases uncertainty:** Token entropy grows monotonically (**0.198 → 0.493 → 0.822**), indicating the model's next-token distribution becomes progressively less peaked as visual evidence degrades.

- **Stable label ≠ stable confidence:** The model outputs the same label (**Tree**) for all conditions, yet **Top-1 probability falls** (**0.969 → 0.907 → 0.804**). This shows logit-based signals can flag reduced reliability even when the predicted answer does not change.

- **Competition between alternatives increases:** The **runner-up probability rises** (**0.018 → 0.048 → 0.082**) while the **Top1–Top2 margin shrinks** (**0.951 → 0.859 → 0.722**), meaning the model is allocating more probability mass to competing tokens (higher ambiguity / higher error risk).

- **Actionable trust decision:** Use **entropy + margin** as a quality gate:
  - **Low-trust / warning** when entropy is high and margin is low (e.g., heavy blur: **entropy=0.822**, **margin=0.722**).
  - **High-trust** when entropy is low and margin is high (e.g., clear image: **entropy=0.198**, **margin=0.951**).

## Stage 2: Expanded VLM Uncertainty Test Plan

| Test Group | Scenario | Prompt | Expected Behavior |
|---|---|---|---|
| Image Quality Degradation | Clear / slight blur / heavy blur | What is the main object in the image? | Entropy should increase as blur increases |
| Object Counting | Easy vs hard counting | How many objects are in the image? | Hard counting should produce higher entropy |
| Semantic Contradiction | Dog image with cat prompt | What breed of cat is this? | Higher uncertainty or flicker |
| Fine-Grained Classification | Common vs rare breed | What specific breed/species is this? | Rare classes should produce higher entropy |
| Spatial Reasoning | Clear vs ambiguous object positions | Is the red ball left or right of the blue box? | Ambiguous layouts should increase entropy |
