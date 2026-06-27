# Security and data boundaries

Putting an LLM in front of company data moves the security boundary, it does not
remove it. The model can be coaxed to reveal context it was given, and retrieval
can surface documents a particular user was never meant to see. Treating the
generative layer as trusted is the most common mistake.

Access control belongs at retrieval, not at generation. Filtering the corpus to
what the requesting user is permitted to see — before passing anything to the
model — keeps unauthorised content out of the context window entirely. Asking the
model to "not reveal" restricted content is not a control; it is a suggestion.

Prompt injection is the corresponding input-side risk: untrusted document text
can carry instructions that hijack the model's behaviour. Defences include
separating instructions from data, constraining the model to answer only from
retrieved context, and validating outputs before they trigger any action.

The validation gate is the last line of defence. Holding anything unverifiable
behind human review means an injected or hallucinated answer cannot silently
become a real decision. Designing for that failure mode is the default, not an
afterthought.
