# Confidence gating

A confidence gate is the trust boundary of a generative system. Before the model
answers, the system checks whether the retrieved evidence is strong enough. When
the top retrieval score falls below a threshold, the system refuses and routes
the question to a human instead of answering on weak evidence.

This encodes a simple principle: AI proposes, a human approves. A wrong answer
delivered confidently is more dangerous than an honest refusal, especially in
security-sensitive or regulated settings where an unverifiable answer that feeds
a real decision is a liability.

The gate is itself a binary classifier — refuse or answer — and so it can and
should be measured. Gate accuracy rewards refusing out-of-corpus questions and
answering in-corpus ones. Precision and recall on the refusal decision expose
the two failure modes separately: refusing answerable questions wastes a human's
time, while answering unanswerable ones leaks ungrounded output.

Setting the threshold is an empirical exercise. Too low and the gate never
trips; too high and the system refuses questions it could have answered. The
right value is found by sweeping it against the evaluation suite.
