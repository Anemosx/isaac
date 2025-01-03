from .q_learner import QLearner
from .coma_learner import COMALearner
from .qtran_learner import QLearner as QTranLearner
from .max_q_learner import MAXQLearner
from .max_q_learner_ddpg import DDPGQLearner
from .max_q_learner_sac import SACQLearner
from .q_learner_w import QLearner as WeightedQLearner
from .qatten_learner import QattenLearner
from .dmaq_qatten_learner import DMAQ_qattenLearner
from .history_q_learner import HistoryQLearner
from .history_dmaq_qatten_learner import HistoryDMAQ_qattenLearner
from .hidden_attacker_learner import HiddenAttackerLearner

REGISTRY = {}

REGISTRY["q_learner"] = QLearner
REGISTRY["w_q_learner"] = WeightedQLearner
REGISTRY["qtran_learner"] = QTranLearner
REGISTRY["coma_learner"] = COMALearner
REGISTRY["max_q_learner"] = MAXQLearner
REGISTRY["sac"] = SACQLearner
REGISTRY["ddpg"] = DDPGQLearner
REGISTRY["qatten_learner"] = QattenLearner
REGISTRY["dmaq_qatten_learner"] = DMAQ_qattenLearner

REGISTRY["history_q_learner"] = HistoryQLearner
REGISTRY["history_dmaq_qatten_learner"] = HistoryDMAQ_qattenLearner

REGISTRY["hidden_attacker_learner"] = HiddenAttackerLearner
