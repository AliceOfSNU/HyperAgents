def get_domain_score_key(domain):
    # Human preferences domains
    if domain in ["search_arena", "paper_review", "imo_grading"]:
        return "overall_accuracy"
    # Balrog game domains
    elif "balrog" in domain:
        return "average_progress"
    # Genesis robotic control domains
    elif "genesis" in domain:
        return "average_fitness"
    # Polyglot domain
    elif "polyglot" in domain:
        return "accuracy_score"
    # IMO proof domain
    elif domain == "imo_proof":
        return "points_percentage"
    # ARC-AGI-3 interactive reasoning domain
    elif domain == "arc_agi3":
        return "average_score"


def get_domain_splits(domain, eval_test=False):
    # Human preferences domains
    if domain in ["search_arena", "paper_review", "imo_grading"]:
        splits = ["train", "val"]
        if eval_test:
            splits.append("test")
        return splits
    # Balrog game domains
    elif "balrog" in domain:
        return ["train"]
    # Genesis robotic control domains
    elif "genesis" in domain:
        return ["train"]
    # Polyglot domain
    elif "polyglot" in domain:
        return ["train"]
    # IMO Proof domain
    elif domain == "imo_proof":
        return ["train"]
    # ARC-AGI-3 interactive reasoning domain
    elif domain == "arc_agi3":
        return ["train"]


def can_domain_ensembled(domain):
    # Human preferences domains
    if domain in ["search_arena", "paper_review"]:
        return True
    # Balrog game domains
    elif "balrog" in domain:
        return False
    # Genesis robotic control domains
    elif "genesis" in domain:
        return False
    # Polyglot domain
    elif "polyglot" in domain:
        return False
    # IMO grading domain
    elif domain == "imo_grading":
        return True
    # IMO proof domain
    elif domain == "imo_proof":
        return False
    # ARC-AGI-3 interactive reasoning domain
    elif domain == "arc_agi3":
        return False


def get_domain_eval_subset(domain):
    # Human preferences domains
    if domain in ["search_arena", "paper_review"]:
        return "_filtered_100_train"
    # Balrog game domains
    elif "balrog" in domain:
        return ""
    # Genesis robotic control domains
    elif "genesis" in domain:
        return ""
    # Polyglot domain
    elif "polyglot" in domain:
        return ""
    # IMO grading domain
    elif domain == "imo_grading":
        return "_filtered_100_train"
    # IMO proof domain
    elif domain == "imo_proof":
        return ""
    # ARC-AGI-3 interactive reasoning domain
    elif domain == "arc_agi3":
        return ""


def get_domain_test_subset(domain):
    # Human preferences domains
    if domain in ["search_arena", "paper_review"]:
        return "_filtered_100_test"
    # Balrog game domains
    elif "balrog" in domain:
        return ""
    # Genesis robotic control domains
    elif "genesis" in domain:
        return ""
    # Polyglot domain
    elif "polyglot" in domain:
        return ""
    # IMO grading domain
    elif domain == "imo_grading":
        return "_filtered_100_test"
    # IMO proof domain
    elif domain == "imo_proof":
        return ""
    # ARC-AGI-3 interactive reasoning domain
    elif domain == "arc_agi3":
        return ""


def get_domain_stagedeval_samples(domain):
    # Human preferences domains
    if domain in ["search_arena", "paper_review"]:
        return 10
    # Balrog game domains
    elif "balrog" in domain:
        return 1
    # Genesis robotic control domains
    elif "genesis" in domain:
        return 3
    # Polyglot domain
    elif "polyglot" in domain:
        return 10
    # IMO grading domain
    elif domain == "imo_grading":
        return 10
    # IMO proof domain
    elif domain == "imo_proof":
        return 10
    # ARC-AGI-3 interactive reasoning domain
    elif domain == "arc_agi3":
        return 1


def get_domain_stagedeval_frac(domain):
    # NOTE: this is hardcoded wrt get_domain_stagedeval_samples and default domain configs
    # Human preferences domains
    if domain in ["search_arena", "paper_review"]:
        return 10/100
    # Balrog game domains
    elif "balrog_babyai" in domain:
        return 1/10
    elif "balrog_minihack" in domain:
        return 1/5
    # Genesis robotic control domains
    elif "genesis" in domain:
        return 3/6
    # Polyglot domain
    elif "polyglot" in domain:
        return 10/60
    # IMO grading domain
    elif domain == "imo_grading":
        return 10/100
    # IMO proof domain
    elif domain == "imo_proof":
        return 10/60
    # ARC-AGI-3 interactive reasoning domain
    # NOTE: assumes a "full" eval of 3 episodes/game; raise --eval_samples to match
    # if you configure more episodes per game in domains/arc_agi3/config/config.yaml
    elif domain == "arc_agi3":
        return 1/3


def has_domain_val_subset(domain):
    # Human preferences domains
    if domain in ["search_arena", "paper_review"]:
        return True
    # Balrog game domains
    elif "balrog" in domain:
        return False
    # Genesis robotic control domains
    elif "genesis" in domain:
        return False
    # Polyglot domain
    elif "polyglot" in domain:
        return False
    # IMO grading domain
    elif domain == "imo_grading":
        return True
    # IMO proof domain
    elif domain == "imo_proof":
        return False
    # ARC-AGI-3 interactive reasoning domain
    elif domain == "arc_agi3":
        return False
