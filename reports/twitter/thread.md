# 🏏 Yorker AI — Psychological Twitter/X Content Strategy

> **The Philosophy:** Twitter/X engagement for technical projects relies on the "Information Gap" and the "Paradox of Honesty." If you claim 90% accuracy, engineers will ignore you. If you claim 53.5% accuracy and explain *why* it's so low (data leakage, strict walk-forward validation), you instantly gain unshakeable credibility. 
> 
> **The Tone:** Sharp, unapologetic, precise. We are not selling a betting tool; we are publishing an open-source manifesto on honest sports analytics.

---

## 📅 Day 1: The Hook & The Hard Truth (April 30)

> **Psychological Goal:** Challenge the status quo. Create an immediate enemy ("cheating models"). Establish extreme credibility through radical honesty. Drive traffic to the dashboard.

**Tweet 1/7 (The Pattern Interrupt Hook)**
```
Every IPL season, Twitter is flooded with ML models claiming 85%+ match prediction accuracy. 

They are almost all lying.

I spent the last week building Yorker AI. The hardest part wasn't the algorithm. It was stopping the model from cheating. 

Here is the truth about cricket ML. 🧵
```

**Tweet 2/7 (The Villain: Data Leakage)**
```
The secret to a 90% accurate model is simple: Data Leakage.

Most devs train models using the playing XI (who is batting/bowling). 
But you don't know the playing XI until the toss—30 minutes before the game. 

If you use post-toss data to claim "pre-match" accuracy, your model is useless in production.
```

**Tweet 3/7 (The Yorker AI Approach)**
```
I wanted to build something brutally honest.

Yorker AI is trained on 1,100+ ball-by-ball IPL matches (2008–2026).
I engineered 27 strict pre-match features:
• Margin-adjusted Elo ratings
• Venue priors (spin vs pace bias)
• Rolling NRR & Momentum

Zero post-toss leakage.
```

**Tweet 4/7 (The Paradox of Honesty)**
```
The results? No sugarcoating.

• Holdout accuracy (IPL 2025): 67.1%
• Walk-forward accuracy (11 seasons): 53.5%
• Best naive baseline: 51.4%

53.5% sounds terrible. But in sports betting, a consistent +2.1% lift over baseline is the difference between going broke and finding real signal.
```

**Tweet 5/7 (The Architecture)**
```
Why I used L1 Logistic Regression instead of XGBoost:

Cricket seasons are short (~74 matches). Complex tree models overfit wildly on small temporal windows.

A recency-weighted L1 Logistic acts as a ruthless feature selector. It drops the noise and only keeps what matters.
```

**Tweet 6/7 (The Reveal)**
```
I deployed the live dashboard today.

No "AI Slop" gradients or soft shadows. Designed like a raw data terminal.
• Live predictions & confidence intervals
• Expected Calibration Error (ECE)
• Monte Carlo championship odds

Play with the data here:
🔗 https://huggingface.co/spaces/Shoryamishra61/yorker-ai
```

**Tweet 7/7 (The Loop)**
```
Tomorrow, I'm dropping the technical deep dive:
→ How I built a custom Elo system for T20 cricket.
→ Why the 2020 COVID season absolutely destroyed the model.
→ What "calibration" means and why binary accuracy is a vanity metric.

Follow to catch part 2. 🏏📊

#IPL2026 #MachineLearning #YorkerAI
```

---

## 📅 Day 2: The Deep Dive (May 1)

> **Psychological Goal:** Prove technical depth. Move from "what" to "how". By showing the messy reality of data engineering (the 2020 COVID anomaly), we prove this isn't just an API wrapper, but actual ground-up research.

**Tweet 1/8 (The Re-Hook)**
```
Yesterday, I launched Yorker AI—an IPL prediction engine built on radical honesty (53.5% accuracy).

Today: The engineering inside the black box. 

How do you compress 4.2 million ball-by-ball records into a single probability?

Thread 🧵👇
```

**Tweet 2/8 (The Foundation)**
```
It starts with the Elo rating system. 

But standard Elo (like in chess) fails in T20 cricket. A 1-run win shouldn't carry the same weight as a 50-run thrashing.

Yorker AI uses a Margin-Adjusted K-factor. Big wins swing momentum harder.
```

**Tweet 3/8 (The Home Fortress)**
```
Elo also ignores geography. In the IPL, the venue is half the battle.

I added a strict +35 Elo boost for the home team. But I also mapped 13 stadiums for:
• Average 1st innings score
• Chase win probability
• Pace vs Spin wicket distribution

Chennai is not Wankhede.
```

**Tweet 4/8 (The 2020 Anomaly)**
```
If you want to humble your model, run a Walk-Forward Validation.

Train on 2016-2019 → Predict 2020. 
The result? Yorker AI hit 41.1% in 2020. Worse than a coin flip.

Why? 
```

**Tweet 5/8 (Context is King)**
```
2020 was the COVID "bubble" season in the UAE. 

• Zero home advantage.
• Venue priors from India were completely invalid.
• Pre-COVID momentum meant nothing.

The model assumed a normal world. The world broke. 
Lesson: ML cannot predict structural paradigm shifts.
```

**Tweet 6/8 (Probability vs Reality)**
```
Let's talk about Calibration. (The metric data scientists hide).

If my model says RCB has a 60% chance to win, and we play that match 100 times, RCB should win exactly 60 times.

Yorker AI's Brier Score is 0.230. 
It knows exactly how uncertain it is.
```

**Tweet 7/8 (Live Call)**
```
Speaking of probability, tonight's match:
RR vs DC at Sawai Mansingh.

Yorker AI says: RR (51.9%).
Confidence: Extreme Toss-up. 

It's essentially a coin flip, driven slightly by RR's home advantage and current NRR differential.
```

**Tweet 8/8 (The CTA)**
```
All the metrics, walk-forward tables, and live probabilities are public on the terminal.

🔗 https://huggingface.co/spaces/Shoryamishra61/yorker-ai

Tomorrow, I'll share the Monte Carlo championship simulation. PBKS fans... you might want to tune in. 

#IPL2026 #DataScience
```

---

## 📅 Day 3: The Simulation & Future (May 2)

> **Psychological Goal:** Shift from technical explanation to actionable, narrative forecasting. People love "what if" scenarios. The Monte Carlo simulation provides an engaging, shareable endpoint to the series.

**Tweet 1/7 (The Narrative Hook)**
```
Part 3 of the Yorker AI series. 

We’ve talked about data leakage, Elo algorithms, and why predicting the IPL is notoriously brutal.

Today: I ran 2,000 Monte Carlo simulations of the rest of the 2026 season. 

Who is actually going to win? 🧵👇
```

**Tweet 2/7 (How Monte Carlo Works)**
```
A Monte Carlo simulation doesn't just predict the next match. 

It takes the current points table, simulates every remaining fixture using Yorker AI's win probabilities, logs the final standings, and repeats it 2,000 times.

It maps the multiverse of the IPL.
```

**Tweet 3/7 (The Top Tier)**
```
The current Title Probability:

🥇 PBKS: 44.8%
🥈 RCB: 25.2%
🥉 RR: 15.6%

Punjab is having a statistical outlier of a season. At 4-0 with a +0.50 NRR, the math heavily favors their momentum. RCB is tracking right behind them.
```

**Tweet 4/7 (The Middle Pack)**
```
The battle for the 4th playoff spot is where the chaos lives.

GT (4.5%), SRH (3.1%), and CSK (3.0%) are statistically gridlocked. 
One bad powerplay, one dew-heavy second innings, and these probabilities invert.
```

**Tweet 5/7 (The Bottom)**
```
The math is ruthless.

DC: 2.2%
LSG: 0.9%
KKR: 0.6%
MI: 0.4%

For MI to win from here, they don't just need to win out; they need specific cascading failures from the top 4. The simulation says it happens 4 times out of 1,000.
```

**Tweet 6/7 (The Open Source Promise)**
```
This ends the launch series, but the engine runs daily.

All data, validation scripts, and the full codebase are open source on GitHub. If you want to fork it, add player-level data, or beat my 53.5% walk-forward baseline—do it.

🔗 https://github.com/Shoryamishra61/Yorker-AI
```

**Tweet 7/7 (Final CTA)**
```
The Yorker AI terminal will update every morning with fresh Elo ratings and new probabilities.

Bookmark the dashboard here: 
🔗 https://huggingface.co/spaces/Shoryamishra61/yorker-ai

If this series changed how you view sports analytics, RT the first tweet. Let's build better models. 🏏

#IPL2026
```

---

## 🧠 Daily Match Framework (Day 4+)

When posting daily predictions, maintain the clinical, terminal-like aesthetic. Do not use hype words.

```text
[YORKER AI // DAILY RUN]

Fixture: [TEAM_A] vs [TEAM_B]
Venue: [Stadium]

Output: [WINNER]
Probability: [XX.X]%
Confidence Matrix: [TIGHT / MODERATE / HIGH]

Delta Drivers:
• [Factor 1, e.g., Venue Spin Bias favors Team A]
• [Factor 2, e.g., Team B rolling NRR deficit]

Live terminal: https://huggingface.co/spaces/Shoryamishra61/yorker-ai
```
