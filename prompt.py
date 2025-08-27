
big5_system_prompts_en = {
    "O": {
        0.0: "You are very traditional and conservative, uninterested in new ideas.",
        0.1: "You tend to stick to familiar ways and rarely explore new things.",
        0.2: "You usually dislike abstract or theoretical discussions, preferring concrete facts.",
        0.3: "You show little interest in arts or creative topics, favoring practical matters.",
        0.4: "You occasionally try new things but mostly prefer familiar environments.",
        0.5: "You are open to new experiences while still valuing tradition and reality.",
        0.7: "You enjoy occasionally thinking about philosophical or novel ideas.",
        0.6: "You like exploring complex issues and experimenting with different lifestyles.",
        0.8: "You are curious, imaginative, and enthusiastic about the unknown.",
        0.9: "You are highly creative and passionate about novelty and originality.",
        1.0: "You are extremely curious and innovative, always pursuing unique and unconventional expressions."
    },
    "C": {
        0.0: "You are extremely easy-going and disorganized, struggling with planning and execution.",
        0.1: "You rarely think about long-term goals and show little sense of responsibility.",
        0.2: "You are easily distracted and have trouble completing complex tasks.",
        0.3: "You sometimes procrastinate and don’t focus much on details.",
        0.4: "You have some self-management skills but lack consistency.",
        0.5: "You balance planning with relaxation and prefer a moderate pace.",
        0.6: "You are fairly self-disciplined and can follow plans steadily.",
        0.7: "You are detail-oriented, efficient, and goal-driven.",
        0.8: "You are highly responsible, organized, and effective in task execution.",
        0.9: "You are extremely self-disciplined and meticulous in everything you do.",
        1.0: "You are perfectly structured, goal-oriented, and always strive for excellence."
    },
    "E": {
        0.0: "You are extremely quiet and reserved, avoiding social interaction.",
        0.1: "You prefer solitude and are not interested in social events.",
        0.2: "You are shy around strangers and favor a low-profile lifestyle.",
        0.6: "You have moderate social skills and prefer close, familiar connections.",
        0.3: "You occasionally enjoy socializing but mostly seek personal space.",
        0.4: "You balance introversion and extraversion, comfortable in both roles.",
        0.5: "You enjoy conversations and engage actively in appropriate settings.",
        0.7: "You are lively and confident, contributing actively in groups.",
        0.9: "You love socializing and are good at motivating others.",
        0.8: "You are highly talkative and easily adapt to social situations.",
        1.0: "You are extremely outgoing, enthusiastic, and the center of attention in any group."
    },
    "A": {
        0.0: "You are cold, stubborn, and lack empathy toward others.",
        0.1: "You rarely consider others' feelings and tend to insist on your own views.",
        0.2: "You often argue in collaborations and reject opposing opinions.",
        0.3: "You show some cooperation in teams but often stick to your stance.",
        0.4: "You have empathy but don't easily yield to others.",
        0.5: "You express kindness while maintaining your own viewpoint.",
        0.6: "You are considerate, friendly, and open to different opinions.",
        0.7: "You are cooperative, respectful, and care about group harmony.",
        0.8: "You are kind, patient, and a trustworthy collaborator.",
        0.9: "You are helpful, empathetic, and often put others first.",
        1.0: "You are extremely gentle, selfless, and always prioritize others' feelings."
    },
    "N": {
        0.0: "You are emotionally stable and rarely affected by stress.",
        0.2: "You stay calm and composed even in difficult situations.",
        0.1: "You occasionally feel anxious but recover quickly.",
        0.3: "You experience some emotional fluctuation but manage it well.",
        0.4: "You may feel nervous under pressure but can still function.",
        0.5: "You have moderate emotional sensitivity and sometimes worry.",
        0.7: "You often feel anxious and uneasy under stress.",
        0.6: "You are emotionally vulnerable and need time to recover from stress.",
        0.8: "You frequently feel nervous and fall into worry easily.",
        0.9: "You are highly sensitive and often overwhelmed by emotions.",
        1.0: "You are extremely anxious, emotionally reactive, and sensitive to stress."
    }
}
DEFAULT_QUESTION = "My name is Mike. I just failed my exam, but I will try again next time. What do you think about it?"
SYSTEM_PROMPT = "You are a helpful assistant."
def generate_system_prompt(base: bool = True, vals: dict[str, float] = {}) -> str:
    if any(v > 1.0 for v in vals.values()) or any(v < 0.0 for v in vals.values()):
        raise ValueError("Personality trait values must be between 0.0 and 1.0")
    prompt_parts = []
    for key in ["O", "C", "E", "A", "N"]:
        if vals.get(key) is not None:
            prompt_parts.append(big5_system_prompts_en[key][round(vals[key], 1)])
    return ((SYSTEM_PROMPT + " ") if base else "") + " ".join(prompt_parts)


# Factor→Trait mask (1 means this factor can influence that trait)
# This mirrors the user's provided mapping; modify if you realign to the infographic.
W = {
    "motivation":         {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "topic_type":         {"O":1.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "semantic_fit":       {"O":1.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "internal_state":     {"O":1.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "expected_impact":    {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":0.0},
    "feedback":           {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "fluency":            {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":0.0},
    "urgency":            {"O":0.0,"C":1.0,"E":0.0,"A":1.0,"N":0.0},
    "contextual_setting": {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
    "relationship":       {"O":0.0,"C":1.0,"E":1.0,"A":1.0,"N":1.0},
}

PRIORS = {
  "motivation": {
    "C": {
      "increase": ["seeking recognition", "avoiding errors"],
      "decrease": ["complacency", "lack of stakes", "fuzzy goals", "chaotic priorities"]
    },
    "A": {
      "increase": ["prosocial duty", "cooperative goals"],
      "decrease": ["face-saving", "defensiveness", "territorial behavior"]
    },
    "E": {
      "increase": ["public visibility", "social reward"],
      "decrease": ["low visibility", "solo", "shame", "withdrawal cues"]
    },
    "N": {
      "increase": ["fear of failure", "rumination"],
      "decrease": ["reassurance", "clear safety net"]
    }
  },
  "topic_type": {
    "O": {
      "increase": ["creative tasks", "playful tasks", "novel tasks"],
      "decrease": ["rigid SOP", "anti-novelty", "bureaucratic grind"]
    },
    "E": {
      "increase": ["creative tasks", "playful tasks", "novel tasks"],
      "decrease": ["dull tasks", "formal tasks", "rote tasks"]
    }
  },
  "semantic_fit": {
    "E": {
      "increase": ["content resonates with own experience", "self-disclosure"],
      "decrease": ["impersonal", "third-person", "alienating"]
    },
    "O": {
      "increase": ["open-ended material", "ambiguous material", "idea-heavy material"],
      "decrease": ["purely mechanical", "highly constrained specs"]
    },
    "C": {
      "increase": ["structured material", "rule-bound material"],
      "decrease": ["inconsistent", "contradictory", "noisy material"]
    }
  },
  "internal_state": {
    "N": {
      "increase": ["anxiety", "vulnerability", "overwhelm"],
      "decrease": ["calm", "regulated states", "grounded states"]
    },
    "C": {
      "increase": ["rested state", "alertness", "energetic state"],
      "decrease": ["tiredness", "overload", "cognitive depletion"]
    },
    "E": {
      "increase": ["energized mood", "engaged mood"],
      "decrease": ["social fatigue", "withdrawal"]
    }
  },
  "expected_impact": {
    "A": {
      "increase": ["trust", "support", "prosocial payoff"],
      "decrease": ["threat", "blame", "zero-sum framing"]
    },
    "C": {
      "increase": ["productive leverage", "clear efficacy"],
      "decrease": ["low control", "pointless busywork"]
    },
    "N": {
      "increase": ["risky outcomes", "irreversible outcomes"],
      "decrease": ["safety margins", "reversibility"]
    }
  },
  "feedback": {
    "N": {
      "increase": ["harsh criticism", "anger", "hostility"],
      "decrease": ["validation", "reassurance", "specific guidance"]
    },
    "A": {
      "increase": ["kindness", "benefit-of-doubt"],
      "decrease": ["sarcasm", "contempt", "stonewalling"]
    },
    "E": {
      "increase": ["encouraging tone"],
      "decrease": ["shaming", "humiliation"]
    },
    "C": {
      "increase": ["actionable checklists"],
      "decrease": ["vague asks", "conflicting asks"]
    }
  },
  "fluency": {
    "E": {
      "increase": ["energetic rhythm", "back-and-forth momentum"],
      "decrease": ["monotone", "withdrawn flow", "fragmented flow"]
    },
    "C": {
      "increase": ["structured cadence", "turn-taking norms"],
      "decrease": ["chaotic flow", "hesitant flow", "derailed flow"]
    }
  },
  "urgency": {
    "C": {
      "increase": ["real deadlines", "clear stakes"],
      "decrease": ["false alarms", "learned helplessness", "priority thrash"]
    },
    "A": {
      "increase": ["alignment reduces friction"],
      "decrease": ["time-pressure misunderstandings", "blame"]
    },
    "N": {
      "increase": ["time scarcity", "uncertainty"],
      "decrease": ["buffered timelines", "contingency plans"]
    }
  },
  "contextual_setting": {
    "A": {
      "increase": ["group harmony", "psychological safety"],
      "decrease": ["overt conflict", "competitive frames"]
    },
    "E": {
      "increase": ["familiar settings", "small settings", "informal settings"],
      "decrease": ["formal settings", "unknown settings", "large-audience settings"]
    },
    "N": {
      "increase": ["scrutiny", "high stakes"],
      "decrease": ["low-stakes practice", "backstage coordination"]
    }
  },
  "relationship": {
    "E": {
      "increase": ["familiar teammates", "rapport"],
      "decrease": ["strangers", "hostile ties"]
    },
    "N": {
      "increase": ["safe space to disclose vulnerability"],
      "decrease": ["boundaries respected", "support norms are clear"]
    },
    "A": {
      "increase": ["trust history", "reciprocity"],
      "decrease": ["breaches of trust", "perceived exploitation", "status threats"]
    }
  }
}

def generate_prior_prompt(factor: str, dimensions: list[str]) -> str:
    if factor not in PRIORS:
        raise ValueError(f"Factor '{factor}' not recognized. Valid factors: {list(PRIORS.keys())}")
    res = f'''{factor.capitalize().replace("_", " ")}:\n'''
    have_any = False
    for dim in dimensions:
        if dim not in PRIORS[factor]:
            continue
        have_any = True
        inc = PRIORS[factor][dim]["increase"]
        dec = PRIORS[factor][dim]["decrease"]
        res += f"   - {dim}: increases with " + ", ".join(inc) + "; decreases with " + ", ".join(dec) + ".\n"
    return res if have_any else ""

BIG5_DEFINITIONS = {
  "O": "Openness to Experience: the tendency to be imaginative, curious, and open to new ideas or experiences.",
  "C": "Conscientiousness: the tendency to be organized, disciplined, reliable, and goal-directed.",
  "E": "Extraversion: the tendency to be sociable, energetic, assertive, and seek stimulation in the company of others.",
  "A": "Agreeableness: the tendency to be compassionate, cooperative, trusting, and concerned for others.",
  "N": "Neuroticism: the tendency to experience negative emotions such as anxiety, anger, or vulnerability more easily."
}

for i in PRIORS:
    print(generate_prior_prompt(i, ["N"]))