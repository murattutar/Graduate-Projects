README — Graduate Projects Portfolio (M.Sc.)
==========================================

Author
------
Murat Tutar

Background
----------
- University of Cologne (Universität zu Köln), M.Sc. (Business Administration) — Supply Chain Management track
- Expected graduation: 2026
- Focus areas: data-driven supply chain decision-making, behavioral operations, optimization, and analytics
- Core toolkit: Python (pandas/numpy), statistical modeling & experimentation, simulation, and mathematical optimization (Gurobi)

Contents (this folder)
----------------------
1) Machine Learning & Predictive Analytics Project.ipynb
2) Prescriptive Analytics & Mathematical Optimization with Gurobi.ipynb
3) Auctions & Dynamic Game Theory - Analytical & Simulation-Based Mathematical Optimization Applications in Strategic Procurement.ipynb
4) Modeling and Analysis of Workload Effect on Fatigue in Gig Delivery - Research Proposal.pdf
5) Giving_Drivers_Autonomy_to_Reorder_Next_Three_Stops.pdf
6) GenAI in Optimal Supplier Selection - Research Proposal.pdf


Project Summaries
=================

1) Machine Learning & Predictive Analytics Project (Jupyter Notebook)
--------------------------------------------------------------------
File: "Machine Learning & Predictive Analytics Project.ipynb"

What it is:
- A multi-part predictive analytics term project combining exploratory analysis and supervised learning.

Key components:
- Exploratory data analysis of an hourly ride-demand dataset (973 days; weather + temporal drivers), including
  missing-hour diagnostics and visualization-based insights (weekday patterns, seasonality proxies, temperature link).
- Predictive maintenance classification with logistic regression:
  - Handles class imbalance (fault vs non-fault) using class weights and decision-threshold tuning.
  - Evaluates with confusion matrix + precision/recall trade-offs and interprets model output.

Methods / topics:
- Feature inspection & correlation checks
- Logistic regression for imbalanced classification
- Thresholding strategy to improve minority-class detection
- Model evaluation (accuracy, precision, recall; confusion matrix)


2) Prescriptive Analytics & Mathematical Optimization with Gurobi (Jupyter Notebook)
-----------------------------------------------------------------------------------
File: "Prescriptive Analytics & Mathematical Optimization with Gurobi.ipynb"

What it is:
- A prescriptive analytics project that formulates and solves a real-world workforce scheduling problem
  (anaesthesiologist/physician staffing) under operational and labor constraints.

Key components:
- Demand pattern analysis and clustering/visual exploration to distinguish weekday vs weekend structure.
- Mixed-integer optimization (Gurobi) to minimize required physicians while meeting hourly coverage,
  plus “always-on” emergency coverage.
- Constraint set includes:
  - 3×8-hour shifts (early/late/night)
  - ≥10 hours rest between shifts
  - ≤40 hours/week, two consecutive days off
  - restricted shift-type combinations per week
  - ≤3 consecutive night shifts
  - cyclic weekly plan consistency (Sunday→Monday continuity)
- Extension beyond deterministic planning:
  - Simulation-based evaluation of undercoverage and utilization under demand variability
  - Stochastic/robust-style discussion of service levels vs staffing cost

Methods / topics:
- Integer programming / MIP modeling
- Schedule feasibility constraints & cyclic planning
- Simulation to stress-test deterministic schedules
- Service level vs cost trade-offs (deterministic vs stochastic framing)


3) Auctions & Dynamic Game Theory — Strategic Procurement (Jupyter Notebook)
---------------------------------------------------------------------------
File: "Auctions & Dynamic Game Theory - Analytical & Simulation-Based Mathematical Optimization Applications in Strategic Procurement.ipynb"

What it is:
- Analytical + simulation-based procurement modeling using auction theory and scoring mechanisms,
  illustrated on a multi-supplier sourcing case (“Vitronium” suppliers).

Key components:
- Cost decomposition and break-even analysis (transport + one-off adjustment costs).
- Monte Carlo simulation of sealed-bid second-price auctions (truthful bidding), including:
  - expected purchase price, supplier win probabilities, conditional price statistics, margins
- Scoring auction extension:
  - compares price-only second-price vs a naïve “second-score” mechanism using a combined score (price + logistics/one-offs)
  - computes expected outcomes via Monte Carlo and numerical quadrature; compares methods
- Demand uncertainty:
  - scenario-weighted total cost of ownership (TCO) across multiple demand levels/probabilities
  - mechanism comparison under uncertainty (cost impact and sensitivity)

Methods / topics:
- Auction simulation (second-price), supplier win probability estimation
- Scoring auctions and expected-value computation
- Monte Carlo vs numerical quadrature comparison
- Scenario analysis and TCO-based mechanism selection


4) Workload → Fatigue → Performance in Gig Delivery (Research Proposal PDF)
--------------------------------------------------------------------------
File: "Modeling and Analysis of Workload Effect on Fatigue in Gig Delivery - Research Proposal.pdf"

What it is:
- Behavioral supply chain research proposal on quick-commerce/gig delivery:
  how cumulative workload drives fatigue and changes performance + self-regulation behavior.

Core research question:
- How does cumulative workload impact behavioral performance (delivery speed / deadline misses)
  and self-regulation (breaks, early log-off) among gig delivery workers?

Hypotheses (high-level):
- H1: Workload increases fatigue (deliveries completed since shift start → higher self-reported fatigue)
- H2: Fatigue deteriorates performance (fatigue → longer completion times)
- H3: Fatigue triggers self-regulation (fatigue → higher likelihood of early log-off / breaks)

Design & analysis plan:
- Laboratory-style simulated delivery task with repeated “deliveries” under time pressure
- Captures fatigue (self-report; with physiological proxy described) and performance measures
- Planned hypothesis testing + regression modeling (incl. behavioral dynamics over the shift)


5) Driver-Led Stop Reordering at DHL (Field Experiment Proposal PDF)
--------------------------------------------------------------------
File: "Giving_Drivers_Autonomy_to_Reorder_Next_Three_Stops.pdf"
Authors: Roman Finger, Serguei Anghelov, Murat Tutar

What it is:
- A proposed 8-week field experiment in last-mile parcel delivery:
  allowing drivers limited autonomy to reorder the next three stops using live traffic context.

Core research question:
- Does limited dynamic reordering (next three stops) improve efficiency without harming customer satisfaction?

Design & outcomes:
- Randomized treatment vs control:
  - Treatment: can reorder next 3 stops (deviate from static planned sequence)
  - Control: follows original route sequence
- Sample plan: ~100–120 drivers in one urban depot; power-informed split
- Outcomes:
  - efficiency: total route duration / productivity
  - service: on-time delivery, first-attempt success, customer satisfaction indicators
- Planned estimation: regression-style treatment effect models with operational controls (route, shift, environment, weather, etc.)


6) GenAI vs Human Judgment in Final Supplier Selection (Research Proposal PDF)
-----------------------------------------------------------------------------
File: "GenAI in Optimal Supplier Selection - Research Proposal.pdf"

What it is:
- A behavioral experiment proposal benchmarking GenAI (LLM-style) recommendations against human decisions
  in the final supplier selection step, with an expert-defined benchmark as ground truth.

Core idea:
- Participants evaluate standardized supplier dossiers (quantitative KPIs + qualitative notes).
- Between-subject design contrasts human vs GenAI as the final decision-maker for the same decision set.

Planned outcomes / hypotheses (high-level):
- Decision quality: alignment with expert benchmark winner / near-optimal set
- Decision efficiency & justification: time to decide and completeness of criterion-based rationale
- Perceived legitimacy: trust, transparency/understanding, acceptance of AI-made decisions

Implementation notes:
- GenAI output is pre-generated and fixed per scenario to avoid run-to-run variance and ensure comparability.


Notes
=====
- These artifacts are course/seminar projects created during graduate study.
- Notebooks contain code + narrative; PDFs contain research proposals and experimental designs.
