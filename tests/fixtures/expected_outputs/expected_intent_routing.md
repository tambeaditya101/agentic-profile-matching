# Expected Intent Routing

The `classify_intent` function (in `src.agent.nodes`) must classify the
following inputs into the listed intents. These are the regression-test
inputs used by `tests/test_agent_flows.py::TestScenario5InterviewQuestions`
and `tests/test_intent_classification.py`.

| Input                                                       | Expected Intent |
|-------------------------------------------------------------|-----------------|
| "Compare the top 3 candidates side by side"                 | `compare`       |
| "What's the difference between Alice and Bob?"              | `compare`       |
| "Why did Alice rank higher than Bob?"                       | `explain`       |
| "Explain the match score for candidate #3"                  | `explain`       |
| "Generate interview questions for Alice"                    | `questions`     |
| "Create a technical assessment for the top candidate"       | `questions`     |
| "Drop the AWS requirement and add TypeScript"               | `refine`        |
| "Add TypeScript as a must-have skill"                       | `refine`        |
| "Make 5 years experience the minimum"                       | `refine`        |
| "Show the full match report for Alice"                      | `report`        |
| "Tell me about candidate #2"                                | `report`        |
| "Done, that's all"                                          | `done`          |
| "Thanks, I'm satisfied"                                     | `done`          |
| "Start over with a new JD"                                  | `new_search`    |
| "Different role, let's search again"                        | `new_search`    |

The keyword-based fallback in `_keyword_intent_classify` handles these
deterministically. The LLM-based classifier should also produce the same
labels but may use slightly different inputs.
