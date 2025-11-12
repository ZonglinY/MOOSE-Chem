import os, re, json, random, time, math
import pandas as pd
from google.genai import types


DISCIPLINE = "chemistry"
# MUTATION_CUSTOM_GUIDE: is added to the prompt to mutate to a novel combination (hypothesis) between research background and an inspiration
MUTATION_CUSTOM_GUIDE = "You should be careful on adopting ML methods as the novel content of the mutation, since currently we are using ML examples to illustrate the derivation of hypothesis from research background and inspirations, and now it seems that the ML concepts can therefore easily be abused. "
# HYPTHESIS_GENERATION_CUSTOM_GUIDE: is added to every prompt involving hypothesis generation
HYPTHESIS_GENERATION_CUSTOM_GUIDE = '''
Please formulate a detailed, valid, feasible, novel, and constructive hypothesis, primarily emphasizing the methodology and mechanistic design. Each step in your hypothesis should be clear, precise, and free from ambiguity. The expected performance or potential impact of the hypothesis is not the main focus and should be mentioned minimally.
The generated hypothesis must not exceed 600 words, but it can be shorter if conciseness doesn't sacrifice essential details (normally 600 words should be more than enough to describe the essential idea and essential details of a hypothesis). The hypothesis must remain concise yet comprehensive, clearly describing all essential chemical components, mechanistic steps, and key technical details, while avoiding unnecessary verbosity or redundant explanations of common scientific knowledge. If your initial hypothesis exceeds 600 words, try to compress it until it meets this constraint without omitting any critical information.
'''


# A collection of prompts for different modules
# more_info: currently only used in additional_round_inspiration_screening, which is a number indicating the number of inspirations to select
def instruction_prompts(module_name, more_info=None):
    if module_name == "first_round_inspiration_screening":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to an impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Usually a paper can be choosed as an inspiration is because it can potentially help to solve or alleviate one problem of a previous method for this research question so that leveraging the concepts related to the inspiration, a better method can be developed based on the previous methods and this inspiration. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. Here the previous method can only inference the multi-layer logistic regression, but can't automatically update its parameters to learn from data. The selected chain rule inspiration can be leveraged to automatically update the parameters in the multi-layer logistic regression, and therefore improve over the previous method to create hypothesis. \nGiven a research question, the background and some of the existing methods for this research question, and several top-tier publications (including their title and abstract), try to identify which publication can potentially serve as an inspiration for the background research question so that combining the research question and the inspiration in some way, a novel, valid, and significant research hypothesis can be formed. Now try to select inspirations based on the background research question. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe potential inspiration candidates are: ", "\n\nNow you have seen the background research question, existing methods, and many potential inspiration candidates. Please carefully think through which publications would be most valuable as inspirations and why, considering how they could help solve problems or limitations in the existing methods.\n\nPlease identify which three literature candidates are the most possible to serve as the inspiration to the background research question. Please reasoning on this task first before providing your answers. Your answer should strictly follow this template:\n\n**Title 1 starts:** [exact title of first selected paper] **Title 1 ends**\n**Title 2 starts:** [exact title of second selected paper] **Title 2 ends**\n**Title 3 starts:** [exact title of third selected paper] **Title 3 ends**"]
    elif module_name == "first_round_inspiration_screening_only_based_on_semantic_similarity":
        prompts = ["You are helping with the scientists to identify the most semantically similar publications. Given a research question, the background and some of the existing methods for this research question, and several top-tier publications (including their title and abstract), try to identify which publication is the most semantically similar to the background research question. Now try to select publications based on background research question. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe potential publication candidates are: ", "\n\nNow you have seen the background research question, and many potential publication candidates. Please carefully analyze the semantic similarity between each candidate and the research question, considering how each publication relates to the research question.\n\nPlease reason through which publications are most semantically similar first before providing your selections. Your answer should strictly follow this template:\n\n**Title 1 starts:** [exact title of first selected paper] **Title 1 ends**\n\n**Title 2 starts:** [exact title of second selected paper] **Title 2 ends**\n\n**Title 3 starts:** [exact title of third selected paper] **Title 3 ends**"]
    elif module_name == "additional_round_inspiration_screening":
        # more_info: args.num_screening_keep_size
        assert isinstance(more_info, int)
        assert more_info > 0
        if more_info > 6:
            print(f"Warning: selecting {more_info} inspirations from all inspiration candidates, is it too much?")
        # might choose more than {num_screening_keep_size} inspirations, also might less than {num_screening_keep_size}
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, a core inspiration in a literature for this research question, and a preliminary research hypothesis from the core inspiration. This hypothesis is aiming for top {DISCIPLINE} venue such as <Nature> or <Science>. You know, to publish a research on Nature or Science, the hypotheis must be novel, valid, and significant enough. ususally it means more than one inspirations should be involved in the hypothesis generation process. Therefore we also have found a series of inspiration candidates, which might provide additional useful information to assist the core inspiration for the next step of hypothesis generation. We have also obtained the potential hypotheses from the combination of each inspiration candidate with the research background question, which might be helpful in determining how each inspiration candidate can potentially contribute to the research question, and whether it could be helpful / complementary to the preliminary hypothesis developed based on the core inspiration. Please help us select around {more_info} inspiration candidates to assist further development of the hypothesis developed from the core inspiration. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe preliminary hypothesis is: ", "\n\nThe potential inspiration candidates and their corresponding hypotheses are: ", f"\n\nNow you have seen the background research question, the core inspiration, the preliminary hypothesis, and the potential inspiration candidates with their corresponding hypotheses. Please carefully analyze which inspiration candidates would best complement the core inspiration, considering how each could enhance or extend the preliminary hypothesis.\n\nPlease reason through which inspirations would best complement the core inspiration first before providing your selections. Your answer should strictly follow this template:\n\n" + "\n\n".join([f"**Title {i} starts:** [exact title of selected paper] **Title {i} ends**" for i in range(1, more_info + 1)])]
    elif module_name == "coarse_hypothesis_generation_only_core_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of conducting research into four steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspiration (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Thirdly it's hypothesis generation based on the background research question and found inspiration; Finally it's about designing and conducting experiments to verify hypothesis. An example is the backpropagation of neural networks. In backpropagation, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. In their paper, the authors have conducted experiments to verify their hypothesis. Now we have identified a good research question, and we have found a core inspiration in a literature for this research question. Please help us generate a novel, valid, and significant research hypothesis based on the background research question and the inspiration. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", f"\n\nNow you have seen the background research question and the core inspiration. Please think carefully about how the inspiration could be leveraged to address limitations or problems in the existing methods, then generate a novel, valid, and significant research hypothesis based on the background research question and the inspiration.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through how to combine the inspiration with the research question. Think step-by-step about the limitations in existing methods and how the inspiration addresses them.\n2. Then, formulate your hypothesis based on your reasoning.\n3. Finally, provide a summary of your key reasoning points.\n\nYour response MUST follow this exact format:\n\n**Hypothesis starts:** [your detailed hypothesis] **Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of the main reasoning steps that led to this hypothesis] **Key Reasoning ends**"]
    elif module_name == "coarse_hypothesis_generation_without_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of conducting research into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly it's hypothesis generation based on the background research question; Finally it's about designing and conducting experiments to verify hypothesis. An example is the backpropagation of neural networks. In backpropagation, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression, and the research hypothesis is the backpropagation itself. In their paper, the authors have conducted experiments to verify their hypothesis. Now we have identified a good research question. Please help us generate a novel, valid, and significant research hypothesis based on the background research question. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", f"\n\nNow you have seen the background research question. Please think carefully about potential approaches to address the research question, considering limitations in existing methods, then generate a novel, valid, and significant research hypothesis.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through how to address the research question. Think step-by-step about the gaps in current methods and potential solutions.\n2. Then, formulate your hypothesis based on your reasoning.\n3. Finally, provide a summary of your key reasoning points.\n\nYour response MUST follow this exact format:\n\n**Hypothesis starts:** [your detailed hypothesis] **Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of the main reasoning steps that led to this hypothesis] **Key Reasoning ends**"]
    # elif module_name == "validness_checking":  # UNUSED
    #     prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be both novel and valid. Here we focus on the validness aspect. Please try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis by any chance is not valid. If not valid, try to give advice on how it could be modified to be more valid. Please directly answer this question. \nThe hypothesis is: \n", "\nPlease carefully analyze the hypothesis for validity first before providing your assessment. Your answer should strictly follow this template:\n\n**Yes or No starts:** [Yes if valid, No if not valid] **Yes or No ends**\n\n**Advice starts:** [your detailed advice for improvement if not valid, or confirmation of validity if valid] **Advice ends**"]
    elif module_name == "novelty_checking":
        prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be novel enough, which means it should not have been proposed by any existing literature before. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more novel. If so, what are your advice to be more novel? Please directly answer this question. Please note that your feedback should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more novel. If so, what are your advice to be more novel?"]
    # elif module_name == "clarity_checking":  # UNUSED
        # prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be clear and specific enough. Please try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific. If so, what are your advice to be more specific? We expect that every detail of the hypothesis is provided, so that a {DISCIPLINE} scientist looking at this hypothesis would have absolutely no doubt on what exactly and comprehensively the hypothesis is in every procedure and in every detail. \
        #            Please directly answer this question. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc. If so, what are your advice to be more specific? (response format: 'Yes or No: \nAdvice:\n')"]
        prompts = [f'''
                    You are assisting {DISCIPLINE} scientists by providing detailed feedback on their newly proposed research hypothesis. The goal is to help them refine it for potential publication in a top {DISCIPLINE} venue such as Nature or Science.

                    As you know, to meet the standards of such venues, a research hypothesis must be clear, specific, and comprehensively detailed.

                    Please carefully evaluate the given hypothesis and answer the following:

                    - Is the hypothesis clear and unambiguous? If not, which parts are vague or confusing?

                    - Does the hypothesis provide sufficient detail on every step, assumption, and condition involved? If not, what additional details would make it more rigorous and specific?

                    - Provide concrete suggestions on how to improve clarity or specificity where needed. When providing the suggestions, you should also ensure that the hypothesis is novel, valid, and significant enough.

                    Your goal is to ensure that any {DISCIPLINE} scientist reading this hypothesis would have absolutely no doubt about its intended meaning, scope, and procedural details.
                   
                    Please directly answer this question. \nThe hypothesis is: \n
                   ''', "\nPlease carefully analyze the hypothesis for clarity and specificity first before providing your assessment. Your answer should strictly follow this template:\n\n**Yes or No starts:** [Yes if needs to be more specific, No if already specific enough] **Yes or No ends**\n\n**Advice starts:** [your detailed advice for improvement if needs more specificity, or confirmation if already specific] **Advice ends**"]
    elif module_name == "four_aspects_checking":
        # prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding. For example, if to introduce a new concept into a method for the hypothesis, the hypothesis shouldn't be only about 'what the new concept is', but 'how specifically the new concept can be leveraged and integrated to the method'. If it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab; (2) novel enough, which means it should not have been proposed by any existing literature before; (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis; (4) significant in research, which means it is more preferable for it to have a relatively significant impact in research community. Currently we don't have resources for real lab experiments, so please try your best to analyze on validness based on your own knowledge and understanding. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? Please directly answer this question. Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, validness, or significance. \
        #            \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant?"]
        prompts = [f'''
                    You are assisting {DISCIPLINE} scientists by providing detailed feedback on their newly proposed research hypothesis. The goal is to help them refine it for potential publication in a top {DISCIPLINE} venue such as Nature or Science.

                    As you know, to meet the standards of such venues, a strong research hypothesis should satisfy the following four criteria:

                    1. Specificity: The hypothesis should provide sufficient methodological detail so that other researchers can clearly understand what the proposed method is and how it will be carried out in practice, leaving no room for confusion or misinterpretation.

                    In particular, if the hypothesis involves introducing a new concept or component into an existing method, it should not stop at describing what the new concept is — it must also explain how the new concept will be concretely integrated, applied, or operationalized within the method.

                    Whenever possible, please also suggest specific parameters, conditions, or operational procedures (e.g., algorithm settings, material properties, experimental setups) that would enable researchers to directly test or implement the hypothesis in a laboratory or experimental environment.

                    2. Novelty: The hypothesis should propose a new idea, mechanism, or approach that has not been reported or established in existing literature.

                    Please carefully assess whether the core idea of the hypothesis — including its key concepts, methods, or combinations of techniques — has already been proposed or widely studied. If any part of the hypothesis appears similar to prior work, please point it out and explain why it may not be sufficiently novel.

                    Conversely, if the hypothesis is novel, please briefly explain what makes it distinct from existing approaches, such as introducing a new principle, a previously unexplored mechanism, or a new combination of known techniques in an original way.

                    3. Validity / Effectiveness: The hypothesis should be testable, verifiable, and practically feasible within real-world {DISCIPLINE} experimental settings.

                    Please evaluate whether the hypothesis can, in principle, be validated through {DISCIPLINE} experiments — assuming sufficient experimental resources. Consider whether the proposed method relies on reasonable assumptions, whether each step is technically executable, and whether the expected outcomes are measurable in a real-world setting.

                    Although we currently do not have access to lab experiments, please assess the validity based on your knowledge and understanding of {DISCIPLINE}, and highlight any potential challenges, limitations, or conditions that may affect the experimental verification of the hypothesis.

                    4. Significance: If possible, the hypothesis should have the potential for meaningful impact in the research community, such as advancing scientific understanding or opening new research directions. It is not necessary for the hypothesis to be groundbreaking, but it should ideally contribute to the field in a way that is recognized as significant by peers.

                    Please provide constructive feedback on whether the given hypothesis meets these four criteria. If any aspect is lacking, please explain why, and suggest concrete ways to improve it.

                    Important: Your feedback should focus on improving the methodological content of the hypothesis — that is, how to make the hypothesis itself more specific, novel, valid, and significant — rather than suggesting ways to improve the writing or description of these qualities.

                    The hypothesis is: \n
                   ''', "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specific, novel, valid, and significant. If so, what are your advice to be more specific, novel, valid, and significant?"]
    elif module_name == "three_aspects_checking_no_significance":
        # prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding. For example, if to introduce a new concept into a method for the hypothesis, the hypothesis shouldn't be only about 'what the new concept is', but 'how specifically the new concept can be leveraged and integrated to the method'. If it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab; (2) novel enough, which means it should not have been proposed by any existing literature before; and (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis. Currently we don't have resources for real lab experiments, so please try your best to analyze on validness based on your own knowledge and understanding. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific, novel, or valid. If so, what are your advice to be more specific, novel, or valid? Please directly answer this question. Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, or validness. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, or valid. If so, what are your advice to be more specific, novel, or valid?"]
        prompts = [f'''
                    You are assisting {DISCIPLINE} scientists by providing detailed feedback on their newly proposed research hypothesis. The goal is to help them refine it for potential publication in a top {DISCIPLINE} venue such as Nature or Science.

                    As you know, to meet the standards of such venues, a strong research hypothesis should satisfy the following four criteria:

                    1. Specificity: The hypothesis should provide sufficient methodological detail so that other researchers can clearly understand what the proposed method is and how it will be carried out in practice, leaving no room for confusion or misinterpretation.

                    In particular, if the hypothesis involves introducing a new concept or component into an existing method, it should not stop at describing what the new concept is — it must also explain how the new concept will be concretely integrated, applied, or operationalized within the method.

                    Whenever possible, please also suggest specific parameters, conditions, or operational procedures (e.g., algorithm settings, material properties, experimental setups) that would enable researchers to directly test or implement the hypothesis in a laboratory or experimental environment.

                    2. Novelty: The hypothesis should propose a new idea, mechanism, or approach that has not been reported or established in existing literature.

                    Please carefully assess whether the core idea of the hypothesis — including its key concepts, methods, or combinations of techniques — has already been proposed or widely studied. If any part of the hypothesis appears similar to prior work, please point it out and explain why it may not be sufficiently novel.

                    Conversely, if the hypothesis is novel, please briefly explain what makes it distinct from existing approaches, such as introducing a new principle, a previously unexplored mechanism, or a new combination of known techniques in an original way.

                    3. Validity / Effectiveness: The hypothesis should be testable, verifiable, and practically feasible within real-world {DISCIPLINE} experimental settings.

                    Please evaluate whether the hypothesis can, in principle, be validated through {DISCIPLINE} experiments — assuming sufficient experimental resources. Consider whether the proposed method relies on reasonable assumptions, whether each step is technically executable, and whether the expected outcomes are measurable in a real-world setting.

                    Although we currently do not have access to lab experiments, please assess the validity based on your knowledge and understanding of {DISCIPLINE}, and highlight any potential challenges, limitations, or conditions that may affect the experimental verification of the hypothesis.

                    Please provide constructive feedback on whether the given hypothesis meets these four criteria. If any aspect is lacking, please explain why, and suggest concrete ways to improve it.

                    Important: Your feedback should focus on improving the methodological content of the hypothesis — that is, how to make the hypothesis itself more specific, novel, and valid — rather than suggesting ways to improve the writing or description of these qualities.

                    The hypothesis is: \n
                   ''', "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specific, novel, and valid. If so, what are your advice to be more specific, novel, and valid?"]
    elif module_name == "four_aspects_checking_and_extra_knowledge":
        prompts = [f"You are assisting {DISCIPLINE} scientists on helping providing feedback to their newly proposed research hypothesis, targetting at publishing the research on a top {DISCIPLINE} venue like Nature or Science. You know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding. For example, if to introduce a new concept into a method for the hypothesis, the hypothesis shouldn't be only about 'what the new concept is', but 'how specifically the new concept can be leveraged and integrated to the method'. If it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab; (2) novel enough, which means it should not have been proposed by any existing literature before; (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis; (4) significant in research, which means it is more preferable for it to have a relatively significant impact in research community. Currently we don't have resources for real lab experiments, so please try your best to analyze on validness based on your own knowledge and understanding. \nPlease try your best to give the {DISCIPLINE} scientists some feedbacks on whether the hypothesis needs to be more specific, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? Please directly answer this question. In addition, if the hypothesis needs some extra knowledge for it to be more complete, valid, or significant in research, please also try to provide (recall) them (if the hypothesis is already complete, it is not necessary to provide external knowledge). Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, validness, or significance. \nThe hypothesis is: \n", "\nPlease give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? In addition, if the hypothesis need some extra knowledge for it to be more complete, valid, or significant in research, please also try to provide (recall) them."]
    elif module_name == "four_aspects_self_numerical_evaluation":
        # prompts = [f"You are known as a diligent and harsh reviewer in {DISCIPLINE} that will spend much time to find flaws when reviewing and therefore usually gives a relatively much lower score than other reviewers. But when you meet with a hypothesis you truly appreciate, you don't mind to give it good scores. Given a not yet peer reviewed research hypothesis in the {DISCIPLINE} domain, try to evaluate the research hypothesis from four research aspects and give score according to evaluation guidelines provided below. All four aspects should be evaluated in a 5 point scale." + f"\nAspect 1: Validness. \n5 points: The hypothesis is a logical next step from current research, strongly supported by theory, perhaps with some indirect experimental evidence or highly predictive computational results. The experimental verification seems straightforward with a high probability of confirming the hypothesis; 4 points: Here, the hypothesis is well-rooted in existing theory with some preliminary data or computational models supporting it. It extends known science into new but logically consistent areas, where experiments are feasible with current technology, and there's a reasonable expectation of positive results; 3 points: This hypothesis is within the realm of theoretical possibility but stretches the boundaries of what's known. It might combine existing knowledge in very novel ways or predict outcomes for which there's no direct evidence yet. There's a conceptual framework for testing, but success is uncertain; 2 points: While the hypothesis might be grounded in some theoretical aspects, it significantly deviates from current understanding or requires conditions or materials that are currently impossible or highly improbable to achieve or synthesize; 1 point: The hypothesis proposes concepts or outcomes that are not only unsupported by current theory but also contradict well-established principles or data. There's no clear path to experimental testing due to fundamental theoretical or practical barriers. " + f"\nAspect 2: Novelty. \n5 points: This level of novelty could fundamentally alter our understanding of {DISCIPLINE} or create entirely new fields. It often involves predictions or discoveries that, if proven, would require a significant overhaul of existing {DISCIPLINE} theories; 4 points: The hypothesis significantly departs from established norms, potentially redefining how certain {DISCIPLINE} phenomena are understood or applied. It might involve entirely new materials or theoretical frameworks; 3 points: This level involves a hypothesis that could potentially lead to new insights or applications. It might challenge minor aspects of current theories or introduce new methodologies or materials; 2 points: The hypothesis introduces a new angle or method within an established framework. It might involve known compounds or reactions but in contexts or combinations not previously explored; 1 point: The hypothesis involves minor tweaks or applications of well-known principles or techniques. It might slightly extend existing knowledge but doesn't introduce fundamentally new concepts. " + f"\nAspect 3: Significance. \n5 points: This hypothesis could fundamentally change one or more branches of {DISCIPLINE}. It might introduce entirely new principles, theories, or methodologies that redefine the boundaries of {DISCIPLINE}; 4 points: This hypothesis challenges current understanding or introduces a concept that could lead to substantial changes in how a particular area of {DISCIPLINE} is viewed or applied. It might lead to new technologies or significant theoretical advancements; 3 points: this hypothesis proposes something new or an innovative approach that could lead to noticeable advancements in a specific area of {DISCIPLINE}. It might open new avenues for research or application but doesn't revolutionize the field; 2 points: This hypothesis might offer a small variation or incremental improvement on existing knowledge. It could potentially refine a known concept but doesn't significantly alter the field; 1 point: The hypothesis addresses a very narrow or already well-established aspect of {DISCIPLINE}. It might confirm what is already known without adding much new insight." + f"\nAspect 4: Potential. \n5 points: The hypothesis, while potentially intriguing now, holds the promise of being revolutionary with the addition of a key methodological component. This could introduce entirely new concepts or fields, fundamentally changing our understanding or capabilities in {DISCIPLINE}; 4 points: The hypothesis, though promising, could be transformative with the right methodological enhancement. This enhancement might lead to groundbreaking discoveries or applications, significantly advancing the field; 3 points: The hypothesis, while interesting in its current form, could be significantly elevated with the right methodological addition. This might lead to new insights or applications that go beyond the initial scope; 2 points: The hypothesis currently offers some value but has the potential for more substantial contributions if enhanced with a new methodological approach. This could lead to incremental advancements in understanding or application; 1 point: The hypothesis, as it stands, might be straightforward or well-trodden. Even with methodological enhancements, it's unlikely to significantly expand current knowledge or applications beyond minor improvements. \
        #            \nThe hypothesis is:\n", "\nPlease give a response to the initial question on scoring the hypothesis from four aspects. Remember that you are a diligent and harsh reviewer. (response format: 'Concise reason for validness score: \nValidness score: \nConcise reason for novelty score: \nNovelty score: \nConcise reason for significance score: \nSignificance score: \nConcise reason for potential score: \nPotential score: \n')."]
        prompts = [f'''
                    You are a harsh and diligent reviewer in {DISCIPLINE}. You are well-known for carefully identifying flaws and usually giving low scores unless a hypothesis is truly exceptional.
                    Given a not-yet-peer-reviewed research hypothesis in {DISCIPLINE}, evaluate it from four aspects: Validness, Novelty, Significance, and Specificity.
                    Each aspect should be scored from 1 to 5 based on the strict guidelines below. High scores (4 or 5) should only be given to truly outstanding hypotheses. Most ordinary or vague hypotheses should receive scores of 1 to 3. Be strict, objective, and critical. Your evaluation should focus only on the content and methodology of the hypothesis, not on writing style.

                    Scoring Guidelines
                    Validness (Objective Soundness)
                    Does the hypothesis make sense based on your knowledge and reasoning ability?
                    5 — Exceptional soundness. Fully coherent and highly reasonable. No weak assumptions.
                    4 — Strong validity. Mostly reasonable with some uncertain assumptions.
                    3 — Barely valid. Possible but weak, speculative, or fragile reasoning.
                    2 — Low validity. Doubtful or inconsistent.
                    1 — Invalid. Contradicts known science or impossible mechanisms.

                    Novelty (Originality)
                    Is the core idea new?
                    5 — Groundbreaking novelty. Fundamentally new principle or mechanism.
                    4 — Highly novel. Significant deviation from existing knowledge.
                    3 — Moderate novelty. New combination but within existing frameworks.
                    2 — Low novelty. Minor variations of existing work.
                    1 — No novelty. Standard or trivial idea.

                    Significance (Research Impact)
                    If true, how much impact does it have?
                    5 — Field-changing. Reshapes core theories or applications.
                    4 — High impact. Advances state-of-the-art or solves key problem.
                    3 — Meaningful. Improves a subfield or opens new directions.
                    2 — Limited. Incremental improvement or niche value.
                    1 — Minimal. Very narrow or little added value.

                    Specificity (Clarity & Methodological Detail)
                    Is the hypothesis detailed and actionable for scientists?
                    5 — Fully detailed. Every key mechanism and parameter is specified.
                    4 — Detailed. Most steps are clear with minor clarification needed.
                    3 — Moderately clear. General methodology but vague key steps.
                    2 — Low specificity. High-level idea only; hard to use directly.
                    1 — Vague. No actionable detail.
                    The hypothesis is:\n
                   ''', '''
Please evaluate the hypothesis from all four aspects. Remember that you are a diligent and harsh reviewer.

First, think through your evaluation carefully, considering each aspect based on the guidelines above.

Then provide your scores. **IMPORTANT: You MUST strictly follow the template format below. Use the exact markers shown, including the "starts:" and "ends" tags:**

**Validness Reason starts:** [Your concise reason for the validness score] **Validness Reason ends**
**Validness Score starts:** [Score from 1-5] **Validness Score ends**

**Novelty Reason starts:** [Your concise reason for the novelty score] **Novelty Reason ends**
**Novelty Score starts:** [Score from 1-5] **Novelty Score ends**

**Significance Reason starts:** [Your concise reason for the significance score] **Significance Reason ends**
**Significance Score starts:** [Score from 1-5] **Significance Score ends**

**Specificity Reason starts:** [Your concise reason for the specificity score] **Specificity Reason ends**
**Specificity Score starts:** [Score from 1-5] **Specificity Score ends**
''']
    elif module_name == "hypothesis_generation_with_feedback_only_core_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question and a core inspiration in a literature for this research question. With them, we have already generated a preliminary coarse-grained research hypothesis. We have also obtain feedbacks on the hypothesis from domain experts in terms of novalty, validity, significance, and clarity. With these feedbacks, please try your best to refine the hypothesis. Please note that during refinement, do not improve a hypothesis's significance by adding expectation of the performance gain of the method or adding description of its potential impact, but you should work on improving the method itself (e.g., by adding or changing details of the methodology). Similar advice for other evaluation aspects (novelty, validness, and clarity), too. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe preliminary hypothesis is: ", "\n\nThe feedbacks from domain experts are: ", f"\n\nNow you have seen the background research question, the core inspiration, the preliminary hypothesis, and the feedbacks from domain experts. Please carefully consider the feedback and think about how to refine the hypothesis accordingly.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through how to address each piece of feedback. Consider which suggestions are most critical and how to incorporate them.\n2. Then, formulate your refined hypothesis based on this analysis.\n3. Finally, provide a summary of how you addressed the feedback.\n\nYour response MUST follow this exact format:\n\n**Refined Hypothesis starts:** [your refined hypothesis incorporating the feedback] **Refined Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of how you addressed the main feedback points] **Key Reasoning ends**"]
    elif module_name == "hypothesis_generation_with_feedback_without_inspiration":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into two steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly it's hypothesis generation based on the background research question. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question. With it, we have already generated a preliminary coarse-grained research hypothesis. We have also obtain feedbacks on the hypothesis from domain experts in terms of novalty, validity, significance, and clarity. With these feedbacks, please try your best to refine the hypothesis. Please note that during refinement, do not improve a hypothesis's significance by adding expectation of the performance gain of the method or adding description of its potential impact, but you should work on improving the method itself (e.g., by adding or changing details of the methodology). Similar advice for other evaluation aspects (novelty, validness, and clarity), too. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe preliminary hypothesis is: ", "\n\nThe feedbacks from domain experts are: ", f"\n\nNow you have seen the background research question, the preliminary hypothesis, and the feedbacks from domain experts. Please carefully consider the feedback and think about how to refine the hypothesis accordingly.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through how to address each piece of feedback. Consider which suggestions are most critical and how to incorporate them.\n2. Then, formulate your refined hypothesis based on this analysis.\n3. Finally, provide a summary of how you addressed the feedback.\n\nYour response MUST follow this exact format:\n\n**Refined Hypothesis starts:** [your refined hypothesis incorporating the feedback] **Refined Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of how you addressed the main feedback points] **Key Reasoning ends**"]
    elif module_name == "hypothesis_generation_mutation_different_with_prev_mutations_only_core_inspiration":
        # Add "In addition, by generating distinct hypothesis, please do not achieve it by simply introducing new concept(s) into the previous hypothesis to make the difference, but please focus on the difference on the methodology of integrating or leveraging the inspiration to give a better answer to the research question (in terms of the difference on the methodology, concepts can be introduced or deleted)."
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, an introduction of previous methods, and a core inspiration in a literature for this research question. The experts know that a proper mixture of these components will definitely lead to a valid, novel, and meaningful research hypothesis. In fact, they already have tried to mix them to compose some research hypotheses (that are supposed to be distinct from each other). Please try to explore a new meaningful way to combine the inspiration with the research background to generate a new research hypothesis that is distinct with all the previous hypotheses in terms of their main method. \
                   {MUTATION_CUSTOM_GUIDE} \
                   The new research hypothesis should ideally be novel, valid, ideally significant, and be enough specific in its methodology. Please note that here we are trying to explore a new meaningful way to leverage the inspiration along with the previous methods (inside or outside the introduction) to better answer the background research question, therefore the new research hypothesis should try to leverage or contain the key information or the key reasoning process in the inspiration, trying to better address the background research question. It means the new research hypothesis to be generated should at least not be completely irrelevant to the inspiration or background research question. In addition, by generating distinct hypothesis, please do not achieve it by simply introducing new concept(s) into the previous hypothesis to make the difference, but please focus on the difference on the methodology of integrating or leveraging the inspiration to give a better answer to the research question  (in terms of the difference on the methodology, concepts can be introduced or deleted). \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe previous hypotheses are: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, and some previous efforts on combining the inspiration with the background for new hypotheses. Please carefully think about how to create a distinct approach that differs from the previous attempts.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through how to create a distinct hypothesis. Analyze what aspects the previous attempts covered and identify unexplored directions.\n2. Then, formulate your hypothesis that explores these new directions.\n3. Finally, provide a summary of why your approach is distinct.\n\nYour response MUST follow this exact format:\n\n**Hypothesis starts:** [your detailed hypothesis that is distinct from previous ones] **Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of why this hypothesis is distinct and how you arrived at it] **Key Reasoning ends**"]
    elif module_name == "final_recombinational_mutation_hyp_gene_same_bkg_insp":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, an introduction of previous methods, and a core inspiration in a literature for this research question. In addition, several experts have already come out of several different hypotheses on how to leverage the inspiration to generate a novel, valid, and significant research hypothesis for the background research question. Please find the bright parts in these hypotheses, leverage the bright parts from them,  modify and combine the good parts of them to generate a better research hypothesis in terms of clarity, novelty, validness, and significance (ideally than any of the given hypotheses). It is not necessary to include methods from every given hypothesis, especially when it is not a good hypothesis. But in general you should try your best to benefit from every given hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypotheses from different expert teams are: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, and the hypotheses from different human scientist teams. Please carefully analyze the strengths of each approach and think about how to combine them effectively.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through the strengths and weaknesses of each hypothesis. Identify which elements are most promising to combine.\n2. Then, formulate your combined hypothesis leveraging the best elements.\n3. Finally, provide a summary of your combination strategy.\n\nYour response MUST follow this exact format:\n\n**Hypothesis starts:** [your refined hypothesis combining the best elements] **Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of which elements you combined and why] **Key Reasoning ends**"]
    elif module_name == "final_recombinational_mutation_hyp_gene_same_bkg_insp_with_feedback":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. \nNow we have identified a good research question, an introduction of previous methods, and a core inspiration in a literature for this research question. In addition, several experts have already come out of several different hypotheses on how to leverage the inspiration to generate a novel, valid, and significant research hypothesis for the background research question. Please find the bright parts in these hypotheses, leverage the bright parts from them,  modify and combine the good parts of them to generate a better research hypothesis in terms of clarity, novelty, validness, and significance (ideally than any of the given hypotheses). It is not necessary to include methods from every given hypothesis, especially when it is not a good hypothesis. But in general you should try your best to benefit from every given hypothesis. In fact, a researcher has already tried to propose hypothesis based on these information, and we have obtained the feedback to his hypothesis, from another respectful researcher. Please try to leverage the feedback to improve the hypothesis, you can leverage all these provided information as your reference. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypotheses from different expert teams are: ", "\n\nThe hypothesis from the researcher is: ", "\n\nThe feedback to the hypothesis from the researcher is: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypotheses from different human scientist teams, the hypothesis from the researcher, and the feedback to the hypothesis from the researcher. Please carefully consider the feedback and think about how to improve the hypothesis.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through the feedback and the various hypotheses. Consider how to address the feedback while leveraging insights from the different approaches.\n2. Then, formulate your refined hypothesis that addresses the feedback.\n3. Finally, provide a summary of your refinement approach.\n\nYour response MUST follow this exact format:\n\n**Refined Hypothesis starts:** [your improved hypothesis incorporating the feedback] **Refined Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of how you addressed the feedback and integrated insights] **Key Reasoning ends**"]
    elif module_name == "final_recombinational_mutation_hyp_gene_between_diff_inspiration":
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. Take backpropagation as an example, the research question is how to use data to automatically improve the parameters of a multi-layer logistic regression with data, the inspiration is the chain rule in mathematics, and the research hypothesis is the backpropagation itself. \nNow we have identified a good research question, an introduction of previous methods, a core inspiration in a literature for this research question, and a hypothesis resulted from leveraging the core inspiration to answer the research background question. This hypothesis is aiming for top {DISCIPLINE} venues such as <Nature> or <Science>. You know, to publish a research on <Nature> or <Science>, the hypotheis must be novel, valid, and significant enough. Ususally it means more than one inspirations should be involved in the hypothesis generation process. Therefore a senior researcher have identified an additional inspiration, along with a hypothesis generated from leveraging the additional inspiration to the research background question. This additional inspiration and its corresponding hypothesis is supposed to provide complementry useful information to assist the further development of the hypothesis developed from the core inspiration. Please find the bright parts in these hypotheses, try to leverage the bright parts from them, modify the hypothesis developed based on the given core inspiration to improve it in terms of novelty, validness, significance, and detailedness. It is not necessary to include methods from every given inspiration & its hypothesis, especially when it is not a good hypothesis. But in general you should try your best to benefit from every given inspiration & its hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nThe hypotheses from other inspirations are: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypothesis from the core inspiration, and the hypotheses resulted from different inspirations. Please carefully analyze how to combine insights from multiple inspirations effectively.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through how different inspirations complement each other. Identify synergies and potential integration points.\n2. Then, formulate your integrated hypothesis that leverages these synergies.\n3. Finally, provide a summary of your integration approach.\n\nYour response MUST follow this exact format:\n\n**Hypothesis starts:** [your integrated hypothesis leveraging multiple inspirations] **Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of how you integrated the multiple inspirations] **Key Reasoning ends**"]
    elif module_name == "final_recombinational_mutation_hyp_gene_between_diff_inspiration_with_feedback":
        prompts = [f"You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into three steps. Firstly it's about the research background, including finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Finally it's hypothesis generation based on the background research question and found inspirations. \nNow we have identified a good research question, an introduction of previous methods, a core inspiration in a literature for this research question, and a hypothesis resulted from leveraging the core inspiration to answer the research background question. This hypothesis is aiming for top {DISCIPLINE} venues such as <Nature> or <Science>. You know, to publish a research on <Nature> or <Science>, the hypotheis must be novel, valid, and significant enough. Ususally it means more than one inspirations should be involved in the hypothesis generation process. Therefore a senior researcher have identified an additional inspiration, along with a hypothesis generated from leveraging the additional inspiration to the research background question. This additional inspiration and its corresponding hypothesis is supposed to provide complementry useful information to assist the further development of the hypothesis developed from the core inspiration. Please find the bright parts in these hypotheses, try to leverage the bright parts from them, modify the hypothesis developed based on the given core inspiration to improve it in terms of novelty, validness, significance, and detailedness. In fact, a researcher has already tried to propose hypothesis based on these information, and we have obtained the feedback to his hypothesis, from another respectful researcher. Please try to leverage the feedback to improve the hypothesis, you can leverage all these provided information as your reference. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nThe hypotheses from other inspirations are: ", "\n\nThe hypothesis from the researcher is: ", "\n\nThe feedback to the hypothesis from the researcher is: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypothesis from the core inspiration, the hypotheses resulted from different inspirations, the hypothesis from the researcher, and the feedback to the hypothesis from the researcher. Please carefully consider all the feedback and insights to create an improved hypothesis.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nIMPORTANT: Please follow this process:\n1. First, reason through the feedback and how the different inspirations can address it. Consider how to create a comprehensive solution.\n2. Then, formulate your refined hypothesis that incorporates all improvements.\n3. Finally, provide a summary of your integration approach.\n\nYour response MUST follow this exact format:\n\n**Refined Hypothesis starts:** [your refined hypothesis incorporating all improvements] **Refined Hypothesis ends**\n\n**Key Reasoning starts:** [a concise summary of how you incorporated feedback and multiple inspirations] **Key Reasoning ends**"]
    elif module_name == "self_extra_knowledge_exploration":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generated a hypothesis for the research question based on the little survey, and the hypothesis generated based on the survey and the inspiration. Although everything goes well now, the hypothesis might only cover one key point (from the inspiration), and might not be complete enough to be a full hypothesis in terms of Validness, Novelty, and Significance. Usually like those papers published on <Nature> or <Science>, a hypothesis could contain two to three key points for it to be enough excellent in terms of Validness, Novelty, and Significance. Please try your best to explore one more knowledge that can potentially improve or complement the existing research hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, and the hypothesis from the core inspiration. Please carefully analyze whether additional knowledge is needed to improve or complement the existing research hypothesis.\n\nPlease reason through whether additional knowledge is needed first before providing your answer. Your answer should strictly follow this template:\n\n**If need extra knowledge starts:** [Yes or No] **If need extra knowledge ends**\n\n**Details starts:** [If No: explain why the hypothesis is complete enough. If Yes: provide the explored additional knowledge that could improve the hypothesis] **Details ends**"]
    elif module_name == "self_extra_knowledge_exploration_with_other_mutations":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generated a hypothesis for the research question based on the little survey, and the hypothesis generated based on the survey and the inspiration. Although everything goes well now, the hypothesis might only cover one key point (from the inspiration), and might not be complete enough to be a full hypothesis in terms of Validness, Novelty, and Significance. Usually like those papers published on <Nature> or <Science>, a hypothesis could contain two to three key points for it to be enough excellent in terms of Validness, Novelty, and Significance. Please try your best to explore one more knowledge that can potentially improve or complement the existing research hypothesis. One more thing to mention, the researchers have already tried to further develop the original hypothesis with extra knowledge, and they have already proposed some potential hypotheses afterwards. Here we want to explore the extra knowledge in a different way with these hypotheses. So please try to develop the original hypothesis with extra knowledge, but not in the same way as any of the hypothesis developed afterwards, so to explore more opportunities. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe original hypothesis from the core inspiration is: ", "\n\nThe hypotheses developed afterwards are: ", "\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the original hypothesis from the core inspiration, and some hypotheses developed afterwards based on the original hypothesis. Please carefully analyze whether additional knowledge is needed that differs from previous attempts.\n\nPlease reason through whether additional knowledge is needed first before providing your answer. Your answer should strictly follow this template:\n\n**If need extra knowledge starts:** [Yes or No] **If need extra knowledge ends**\n\n**Details starts:** [If No: explain why the hypothesis is complete enough. If Yes: provide the explored additional knowledge that differs from previous attempts] **Details ends**"]
    elif module_name == "hypothesis_generation_with_extra_knowledge":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generated a hypothesis for the research question based on the little survey, and the hypothesis generated based on the survey and the inspiration. Although everything goes well now, the hypothesis might only cover one key point (from the inspiration), and might not be complete enough to be a full hypothesis in terms of Validness, Novelty, and Significance. Usually like those papers published on <Nature> or <Science>, a hypothesis could contain two to three key points for it to be enough excellent in terms of Validness, Novelty, and Significance. Therefore the researcher has already explored the additional knowledge to make the hypothesis more complete. Please try your best to generate a new hypothesis based on the background research question, the inspiration, the additional knowledge, and the given preliminary hypothesis. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe hypothesis from the core inspiration is: ", "\n\nThe additional knowledge is: ", f"\n\nNow you have seen the background research question, an introduction of the previous methods, the core inspiration, the hypothesis from the core inspiration, and the additional knowledge. Please carefully think about how to integrate the additional knowledge to improve the hypothesis.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nPlease reason through how to integrate the additional knowledge first before providing your hypothesis. Your answer should strictly follow this template:\n\n**Hypothesis starts:** [your enhanced hypothesis incorporating the additional knowledge] **Hypothesis ends**"]
    # here with_extra_knowledge" means the hypothesis is generated based on the core inspiration and the extra knowledge, but not that the feedback need to cover extra knowledge
    elif module_name == "provide_feedback_to_hypothesis_four_aspects_with_extra_knowledge":
        prompts = [f"You are helping to develop a {DISCIPLINE} research hypothesis. A senior researcher has identified the research question, a little survey on the background of the research question, a key inspiration paper used to generate a hypothesis for the research question based on the little survey, an extra knowledge that should be usedful to develop a hypothesis, and the hypotheses developed based on the inspiration and the extra knowledge. Please try to give some feedback to the research hypothesis. Specifically, you know, to publish a research on Nature or Science, the hypothesis must be (1) specific enough, which means the research hypothesis should contain enough details of the method for the researchers to know at least what the method is without any confusion or misunderstanding (if it is within your ability, please also provide details on the parameters of the hypothesis, so that the researchers can directly test the hypothesis in their lab); (2) novel enough, which means it should not have been proposed by any existing literature before; (3) completely valid, which means a real {DISCIPLINE} experiments should be able to verify the hypothesis; (4) significant in research, which means it is more preferable for it to have a relatively significant impact in research community. \nPlease try your best to give the senior researcher some feedbacks on whether the hypothesis needs to be more specific, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant? Please directly answer this question. Please note that your feedback to these aspects should focus on the methodology in the hypothesis, but not how to add descriptions of its novelty, significance, or validness. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe extra knowledge is: ", "\n\nThe hypothesis is: ", "\n\nNow you have seen the background research question, the core inspiration, the extra knowledge, and the hypothesis. Please give a response to the initial question on determining whether the research hypothesis need to be more specifc, novel, valid, or significant. If so, what are your advice to be more specific, novel, valid, or significant?"]
    elif module_name == "hypothesis_refinement_with_feedback_with_extra_knowledge":
        prompts = ["You are helping with the scientific hypotheses generation process. We in general split the period of research hypothesis proposal into four steps. Firstly it's about finding a good and specific background research question, and an introduction of the previous methods under the same topic; Secondly its about finding inspirations (mostly from literatures), which combined with the background research question, can lead to a impactful research hypothesis; Thirdly it's about finding extra knowledge that work along with the inspiration can lead to a more complete hypothesis. Finally it's hypothesis generation based on the background research question, the found inspirations, and the extra knowledge. \nNow we have identified a good research question, a core inspiration in a literature for this research question, and extra knowledge. With them, we have already generated a preliminary research hypothesis. We have also obtain feedbacks on the hypothesis from domain experts in terms of novalty, validity, significance, and clarity. With these feedbacks, please try your best to refine the hypothesis. Please note that during refinement, do not improve a hypothesis's significance by adding expectation of the performance gain of the method or adding description of its potential impact, but you should work on improving the method itself (e.g., by adding or changing details of the methodology). Similar advice for other evaluation aspects (novelty, validness, and clarity), too. \nThe background research question is: ", "\n\nThe introduction of the previous methods is:", "\n\nThe core inspiration is: ", "\n\nThe extra knowledge is: ", "\n\nThe preliminary hypothesis is: ", "\n\nThe feedbacks from domain experts are: ", f"\n\nNow you have seen the background research question, the core inspiration, the extra knowledge, the preliminary hypothesis, and the feedbacks from domain experts. Please carefully consider the feedback and think about how to refine the hypothesis accordingly.\n\n{HYPTHESIS_GENERATION_CUSTOM_GUIDE}\n\nPlease reason through how to address the feedback first before providing your refined hypothesis. Your answer should strictly follow this template:\n\n**Refined Hypothesis starts:** [your refined hypothesis incorporating the feedback] **Refined Hypothesis ends**"]
    elif module_name == "eval_matched_score":
        prompts = [f"You are helping to evaluate the quality of a proposed research hypothesis in {DISCIPLINE} by a phd student. The groundtruth hypothesis will also be provided to compare. Here we mainly focus on whether the proposed hypothesis has covered the key points in terms of the methodology in the groundtruth hypothesis. You will also be given a summary of the key points in the methodology of the groundtruth hypothesis for reference. Please note that for the proposed hypothesis to cover one key point, it is not necessary to explicitly mention the name of the key point, but might also can integrate the key point implicitly in the proposed method. The evaluation criteria is called 'Matched score', which is in a 6-point Likert scale (from 5 to 0). Particularly, 5 points mean that the proposed hypothesis (1) covers all the key points and leverage them similarly as in the methodology of the groundtruth hypothesis, and (2) does not contain any extra key point that has apparent flaws; 4 points mean that the proposed hypothesis (1) covers all the key points (or at least three key points) and leverage them similarly as in the methodology of the groundtruth hypothesis, (2) but also with extra key points that have apparent flaws; 3 points mean that the proposed hypothesis (1) covers at least two key points and leverage them similarly as in the methodology of the groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, (3) might or might not contain extra key points; 2 points mean that the proposed hypothesis (1) covers at least one key point in the methodology of the groundtruth hypothesis, and leverage it similarly as in the methodology of groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, and (3) might or might not contain extra key points; 1 point means that the proposed hypothesis (1) covers at least one key point in the methodology of the groundtruth hypothesis, (2) but is used differently as in the methodology of groundtruth hypothesis, and (3) might or might not contain extra key points; 0 point means that the proposed hypothesis does not cover any key point in the methodology of the groundtruth hypothesis at all. Please note that the total number of key points in the groundtruth hypothesis might be less than three, so that multiple points can be given. E.g., there's only one key point in the groundtruth hypothesis, and the proposed hypothesis covers the one key point, it's possible to give 2 points, 4 points, and 5 points. In this case, we should choose score from 4 points and 5 points, depending on the existence and quality of extra key points. 'Leveraging a key point similarly as in the methodology of the groundtruth hypothesis' means that in the proposed hypothesis, the same (or very related) concept (key point) is used in a similar way with a similar goal compared to the groundtruth hypothesis (not necessarily for the proposed hypothesis to be exactly the same with the groudtruth hypothesis to be classified as 'similar'). When judging whether an extra key point has apparent flaws, you should use your own knowledge to judge, but rather than to rely on the count number of pieces of extra key point to judge. \nPlease evaluate the proposed hypothesis based on the groundtruth hypothesis. \nThe proposed hypothesis is: ", "\n\nThe groundtruth hypothesis is: ", "\n\nThe key points in the groundtruth hypothesis are: ", "\n\nPlease carefully reason through your evaluation based on the criteria above first before providing your score. Your answer should strictly follow this template:\n\n**Matched score starts:** [0-5] **Matched score ends**"]
    # elif module_name == "eval_matched_score_hard":  # UNUSED
    #     prompts = ["You are helping to evaluate the quality of a proposed research hypothesis by a phd student. The groundtruth hypothesis will also be provided to compare. Here we mainly focus on whether the proposed hypothesis has covered the key points of the groundtruth hypothesis. You will also be given a summary of the key points in the groundtruth hypothesis for reference. The evaluation criteria is called 'Matched score', which is in a 6-point Likert scale (from 5 to 0). Particularly, \n5 points mean that the proposed hypothesis (1) covers three key points (or covers all the key points) in the groundtruth hypothesis, where every key point is leveraged nearly identically as in the groundtruth hypothesis, and (2) does not contain any extra key point(s) that is redundant, unnecessary, unhelpful, or harmful; \n4 points mean that the proposed hypothesis (1) covers three key points (or covers all the key points) in the groundtruth hypothesis, where every key point is leveraged nearly identically as in the groundtruth hypothesis, and (2) but also contain extra key point(s) that is redundant, unnecessary, unhelpful, or harmful; \n3 points mean that the proposed hypothesis (1) covers two key points in the groundtruth hypothesis, where every key point is leveraged nearly identically as in the groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, and (3) might or might not contain extra key points; \n2 points mean that the proposed hypothesis (1) covers one key point in the groundtruth hypothesis, and leverage it nearly identically as in the groundtruth hypothesis, (2) but does not cover all key points in the groundtruth hypothesis, and (3) might or might not contain extra key points; \n1 point means that the proposed hypothesis (1) covers at least one key point in the groundtruth hypothesis, but all the covered key point(s) are used differently as in the groundtruth hypothesis, and (2) might or might not contain extra key points; \n0 point means that the proposed hypothesis does not cover any key point in the groundtruth hypothesis at all. \nUsually total the number of key points a groundtruth hypothesis contain is less than or equal to three. Please note that the total number of key points in the groundtruth hypothesis might be less than three, so that multiple points can be given. E.g., there's only one key point in the groundtruth hypothesis, and the proposed hypothesis covers the one key point nearly identically, it's possible to give 2 points, 4 points, and 5 points. In this case, we should choose score from 4 points and 5 points, depending on the existence and quality of extra key points. 'Leveraging a key point nearly identically as in the groundtruth hypothesis means that in the proposed hypothesis, the same (or very related) concept (key point) is used in a very similar way with a very similar goal compared to the groundtruth hypothesis. \nWhen judging whether an extra key point has apparent flaws, you should use your own knowledge and understanding of that discipline to judge, rather than only relying on the count number of pieces of extra key point to judge. \nPlease evaluate the proposed hypothesis based on the groundtruth hypothesis. \nThe proposed hypothesis is: ", "\n\nThe groundtruth hypothesis is: ", "\n\nThe key points in the groundtruth hypothesis are: ", "\n\nPlease carefully reason through your evaluation based on the criteria above first before providing your score. Your answer should strictly follow this template:\n\n**Matched score starts:** [0-5] **Matched score ends**"]
    else:
        raise NotImplementedError
    
    return prompts


# Input:
#   input_list: [[item0, item1], [item0, item1], ...] OR [item0, item1]
# Output:
#   output_list: [[item1, item0], [item1, item0], ...] OR [item1, item0]
#   The order of the items in the input_list is reversed.
#   If the input_list is a list of lists, the order of the items in each list is reversed.
#   If the input_list is a list of strings, the order of the items in the list is reversed.
def exchange_order_in_list(input_list):
    output_list = []
    for cur_input_list in input_list:
        if isinstance(cur_input_list, list):
            assert len(cur_input_list) == 2
            output_list.append(cur_input_list[::-1])
        elif isinstance(cur_input_list, str):
            assert len(input_list) == 2
            output_list = input_list[::-1]
            break
        else:
            raise ValueError("Invalid input type. Expected list or string.")
    return output_list



# # UNUSED: calculate the ratio if how the selected inspirations hit the groundtruth inspirations. 
# def calculate_average_ratio_top1_top2(file_dir):
#     with open(file_dir, 'r') as f:
#         d = json.load(f)
# 
#     ratio_top1, ratio_top2 = 0, 0
#     cnt_ratio = 0
#     for i in d[1]:
#         cur_ratio = d[1][i]
#         ratio_top1 += cur_ratio[0]
#         ratio_top2 += cur_ratio[1]
#         cnt_ratio += 1
#     ratio_top1 = ratio_top1 / cnt_ratio
#     ratio_top2 = ratio_top2 / cnt_ratio
#     return ratio_top1, ratio_top2


## Function: used by load_chem_annotation() and load_chem_annotation_with_feedback(); used to recover background_survey_strict and background_question_strict
# background_strict_raw: a list of the raw background survey, some of them are "NA"; when it is "NA", we should find its component in background_normal
# background_normal: a list of the normal background survey, no "NA"
# background_strict_raw_nan_indicator: a list of boolean values indicating whether the corresponding background_strict_raw is "NA"
def recover_raw_background(background_strict_raw, background_normal, background_strict_raw_nan_indicator):
    background_strict = []
    for cur_survey_id, cur_survey in enumerate(background_strict_raw):
        if background_strict_raw_nan_indicator[cur_survey_id]:
            cur_value = background_normal[cur_survey_id].strip()
            background_strict.append(cur_value)
        else:
            cur_survey = cur_survey.strip()
            # this assertion is to make sure the content is not variants of "NA"
            assert len(cur_survey) > 10
            cur_value = cur_survey
            background_strict.append(cur_value)
    return background_strict
    


# load xlsx annotations, bkg question -> inspirations
# bkg_q: [bq0, bq1, ...]
# dict_bkg2insp: {'bq0': [insp0, insp1, ...], 'bq1': [insp0, insp1, ...], ...}
# dict_bkg2survey: {'bq0': survey0, 'bq1': survey1, ...}
def load_chem_annotation(chem_annotation_path, if_use_strict_survey_question, if_use_background_survey=1):
    assert if_use_strict_survey_question in [0, 1]
    assert if_use_background_survey in [0, 1]
    if if_use_background_survey == 0:
        print("Warning: Not Using Survey.")
    ## load chem_research.xlsx to know the groundtruth inspirations
    chem_annotation = pd.read_excel(chem_annotation_path, 'Overall')
    nan_values = chem_annotation.isna()
    bkg_survey = list(chem_annotation[chem_annotation.columns[4]])
    # some of the components are "NA"; if it is NA, we should find its component in bkg_survey
    bkg_survey_strict_raw = list(chem_annotation[chem_annotation.columns[5]])
    # print("bkg_survey_strict_raw: ", bkg_survey_strict_raw)
    bkg_survey_strict = recover_raw_background(bkg_survey_strict_raw, bkg_survey, nan_values[chem_annotation.columns[5]])
    bkg_q = list(chem_annotation[chem_annotation.columns[6]])
    # some of the components are "NA"; if it is NA, we should find its component in bkg_q
    bkg_q_strict_raw = list(chem_annotation[chem_annotation.columns[7]])
    bkg_q_strict = recover_raw_background(bkg_q_strict_raw, bkg_q, nan_values[chem_annotation.columns[7]])
    insp1 = list(chem_annotation[chem_annotation.columns[9]])
    insp2 = list(chem_annotation[chem_annotation.columns[11]])
    insp3 = list(chem_annotation[chem_annotation.columns[13]])
    groundtruthHyp = list(chem_annotation[chem_annotation.columns[15]])
    reasoningprocess = list(chem_annotation[chem_annotation.columns[17]])
    note = list(chem_annotation[chem_annotation.columns[18]])
    ## determine which version of survey and question to use
    if if_use_strict_survey_question:
        bkg_survey = bkg_survey_strict
        bkg_q = bkg_q_strict
    ## start looping for collection
    dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2reasoningprocess = {}, {}, {}, {}, {}
    dict_bkg2idx, dict_idx2bkg = {}, {}
    for cur_b_id, cur_b in enumerate(bkg_q):
        # update bkg_q to remove leading and trailing spaces
        cur_b = cur_b.strip()
        bkg_q[cur_b_id] = cur_b
        ## dict_bkg2insp
        cur_b_insp = []
        # insp1
        if nan_values[chem_annotation.columns[9]][cur_b_id] == False:
            cur_b_insp.append(insp1[cur_b_id].strip())
        # insp2
        if nan_values[chem_annotation.columns[11]][cur_b_id] == False:
            cur_b_insp.append(insp2[cur_b_id].strip())
        # insp3
        if nan_values[chem_annotation.columns[13]][cur_b_id] == False:
            cur_b_insp.append(insp3[cur_b_id].strip())
        dict_bkg2insp[cur_b] = cur_b_insp
        ## dict_bkg2survey
        if if_use_background_survey:
            assert nan_values[chem_annotation.columns[4]][cur_b_id] == False
            dict_bkg2survey[cur_b] = bkg_survey[cur_b_id].strip()
        else:
            dict_bkg2survey[cur_b] = "Survey not provided. Please overlook the survey."
        ## dict_bkg2groundtruthHyp
        assert nan_values[chem_annotation.columns[15]][cur_b_id] == False
        dict_bkg2groundtruthHyp[cur_b] = groundtruthHyp[cur_b_id].strip()
        ## dict_bkg2reasoningprocess
        assert nan_values[chem_annotation.columns[17]][cur_b_id] == False
        dict_bkg2reasoningprocess[cur_b] = reasoningprocess[cur_b_id].strip()
        ## dict_bkg2note
        assert nan_values[chem_annotation.columns[18]][cur_b_id] == False
        dict_bkg2note[cur_b] = note[cur_b_id].strip()
        ## dict_bkg2idx, dict_idx2bkg
        dict_bkg2idx[cur_b] = cur_b_id
        dict_idx2bkg[cur_b_id] = cur_b
    return bkg_q, dict_bkg2insp, dict_bkg2survey, dict_bkg2groundtruthHyp, dict_bkg2note, dict_bkg2idx, dict_idx2bkg, dict_bkg2reasoningprocess


# load xlsx annotations and data id, return the background question and inspirations; used for check_moosechem_output() in analysis.py
def load_bkg_and_insp_from_chem_annotation(chem_annotation_path, background_question_id, if_use_strict_survey_question):
    # load chem_research.xlsx to know the groundtruth inspirations
    chem_annotation = pd.read_excel(chem_annotation_path, 'Overall')
    nan_values = chem_annotation.isna()
    # bkg_survey = list(chem_annotation[chem_annotation.columns[4]])
    bkg_q = list(chem_annotation[chem_annotation.columns[6]])
    bkg_q_strict_raw = list(chem_annotation[chem_annotation.columns[7]])
    bkg_q_strict = recover_raw_background(bkg_q_strict_raw, bkg_q, nan_values[chem_annotation.columns[7]])
    insp1 = list(chem_annotation[chem_annotation.columns[9]])
    insp2 = list(chem_annotation[chem_annotation.columns[11]])
    insp3 = list(chem_annotation[chem_annotation.columns[13]])
    # whether use strict version of bkg_q
    if if_use_strict_survey_question:
        bkg_q = bkg_q_strict

    cur_bkg = bkg_q[background_question_id].strip()
    cur_insp_list = []
    # insp1
    if nan_values[chem_annotation.columns[9]][background_question_id] == False:
        cur_insp_list.append(insp1[background_question_id].strip())
    # insp2
    if nan_values[chem_annotation.columns[11]][background_question_id] == False:
        cur_insp_list.append(insp2[background_question_id].strip())
    # insp3
    if nan_values[chem_annotation.columns[13]][background_question_id] == False:
        cur_insp_list.append(insp3[background_question_id].strip())
    return cur_bkg, cur_insp_list

    

# load the title and abstract of the groundtruth inspiration papers and random high-quality papers
# INPUT
#   title_abstract_collector_path: the file path of the inspiration corpus
#       It should contain a list of [title, abstract] pairs: [[title, abstract], ...]
# OUTPUT
#   title_abstract_collector: [[title, abstract], ...]
#   dict_title_2_abstract: {'title': 'abstract', ...}
def load_dict_title_2_abstract(title_abstract_collector_path):
    ## load title_abstract_collector
    with open(title_abstract_collector_path, 'r') as f:
        # title_abstract_collector: [[title, abstract], ...]
        title_abstract_collector = json.load(f)
    print("Number of title-abstract pairs loaded: ", len(title_abstract_collector))
    ## Transfer title_abstract_collector to dict_title_2_abstract
    # dict_title_2_abstract: {'title': 'abstract', ...}
    dict_title_2_abstract = {}
    for cur_item in title_abstract_collector:
        if cur_item[0] in dict_title_2_abstract:
            # print("Warning: seen before: ", cur_item[0])
            continue
        dict_title_2_abstract[cur_item[0]] = cur_item[1]
    return title_abstract_collector, dict_title_2_abstract


# inspiration_path: path to selected inspiration, eg, "coarse_inspiration_search_gpt4.json"
# load coarse-grained / fine-grained inspiration screening results
## Output
# organized_insp: {'bq': [[title, reason], [title, reason], ...]}
def load_found_inspirations(inspiration_path, idx_round_of_first_step_insp_screening):
    with open(inspiration_path, 'r') as f:
        selected_insp_info = json.load(f)
    # organized_insp: {'bq': [screen_results_round1, screen_results_round2, ...], ...}
    #   screen_results_round1: [[title, reason], [title, reason], ...]
    organized_insp = selected_insp_info[0]
    organized_insp_hit_ratio = selected_insp_info[1]
    # dict_bkg_insp2idx: {'bq': {'title': idx, ...}, ...}
    # dict_bkg_idx2insp: {'bq': {idx: 'title', ...}, ...}
    dict_bkg_insp2idx, dict_bkg_idx2insp = {}, {}
    # organized_insp_selected_round: {'bq': [[title, reason], [title, reason], ...]}
    organized_insp_selected_round = {}
    for bq in organized_insp:
        dict_bkg_insp2idx[bq] = {}
        dict_bkg_idx2insp[bq] = {}
        organized_insp_selected_round[bq] = []
        for idx, cur_insp in enumerate(organized_insp[bq][idx_round_of_first_step_insp_screening]):
            dict_bkg_insp2idx[bq][cur_insp[0]] = idx
            dict_bkg_idx2insp[bq][idx] = cur_insp[0]
            organized_insp_selected_round[bq].append(cur_insp)
        print("\nNumber of inspirations loaded: {} for background question: {}".format(len(organized_insp_selected_round[bq]), bq))
    return organized_insp_selected_round, dict_bkg_insp2idx, dict_bkg_idx2insp


## Input
# bkg_q: text
# dict_bkg2insp: {'bq0': [insp0, insp1, ...], 'bq1': [insp0, insp1, ...], ...}
## Output
# organized_insp: {'bq': [[title, reason], [title, reason], ...]}
# dict_bkg_insp2idx: {'bq': {'title': idx, ...}, ...}
# dict_bkg_idx2insp: {'bq': {idx: 'title', ...}, ...}
def load_groundtruth_inspirations_as_screened_inspirations(bkg_q, dict_bkg2insp):
    # organized_insp
    organized_insp = {}
    organized_insp[bkg_q] = []
    # dict_bkg_insp2idx, dict_bkg_idx2insp
    dict_bkg_insp2idx, dict_bkg_idx2insp = {}, {}
    dict_bkg_insp2idx[bkg_q] = {}
    dict_bkg_idx2insp[bkg_q] = {}
    # iterating through the inspirations
    gdth_insps = dict_bkg2insp[bkg_q]
    for cur_insp_id, cur_insp in enumerate(gdth_insps):
        organized_insp[bkg_q].append([cur_insp, "Not provided yet."])
        dict_bkg_insp2idx[bkg_q][cur_insp] = cur_insp_id
        dict_bkg_idx2insp[bkg_q][cur_insp_id] = cur_insp
    return organized_insp, dict_bkg_insp2idx, dict_bkg_idx2insp



## Input
# selected_insp: {'bq': [screen_results_round1, screen_results_round2, ...], ...}
#   screen_results_round1: [[[title, reason], [title, reason]], [[title, reason], [title, reason]], ...]
## Output
# organized_insp: {'bq': [screen_results_round1_org, screen_results_round2_org, ...]}
#   screen_results_round1_org: [[title, reason], [title, reason], ...]
def organize_raw_inspirations(selected_insp):
    # organized_insp: {'bq': [[title, reason], [title, reason], ...]}
    organized_insp = {}
    for bq in selected_insp:
        assert bq not in organized_insp
        organized_insp[bq] = []
        # cur_screen_results_round: [[[title, reason], [title, reason]], [[title, reason], [title, reason]], ...]
        for cur_round_id, cur_screen_results_round in enumerate(selected_insp[bq]):
            organized_insp[bq].append([])
            # round_insp: [[title, reason], [title, reason]] (most likely only two or three inspirations)
            for round_insp in cur_screen_results_round:
                organized_insp[bq][cur_round_id] += round_insp
    return organized_insp


# # UNUSED: insp_grouping_results: {insp title: [[other insp title, reason], ...]}
# def load_grouped_inspirations(inspiration_group_path):
#     with open(inspiration_group_path, 'r') as f:
#         insp_grouping_results = json.load(f)
#     return insp_grouping_results


# # UNUSED: coarse_grained_hypotheses: {core_insp_title: [[hypothesis, reasoning process], ...]}
# def load_coarse_grained_hypotheses(coarse_grained_hypotheses_path):
#     with open(coarse_grained_hypotheses_path, 'r') as f:
#         coarse_grained_hypotheses = json.load(f)
#     return coarse_grained_hypotheses
    

# Call Openai API,k input is prompt, output is response
def llm_generation(prompt, model_name, client, temperature=1.0, api_type=0):
    # print("prompt: ", prompt)
    if "claude-3-haiku" in model_name:
        max_tokens = 4096
    else:
        max_tokens = 8192
    cnt_max_trials = 1
    # start inference util we get generation
    for cur_trial in range(cnt_max_trials):
        try:
            if api_type in [0, 1]:
                completion = client.chat.completions.create(
                model=model_name,
                temperature=temperature,
                max_tokens=max_tokens,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": prompt}
                    ]
                )
                generation = completion.choices[0].message.content.strip()
            # google client
            elif api_type == 2:
                response = client.models.generate_content(
                    model=model_name, 
                    contents=prompt,
                    config=types.GenerateContentConfig(
                        thinking_config=types.ThinkingConfig(thinking_budget=0)
                    )
                )
                generation = response.text.strip()
            else:
                raise NotImplementedError
            break
        except Exception as e:
            print("API Error occurred: ", e)
            time.sleep(0.25)
            if cur_trial == cnt_max_trials - 1:
                raise Exception("Failed to get generation after {} trials because of API error: {}.".format(cnt_max_trials, e))
    # print("generation: ", generation)
    return generation


"""
UNUSED FUNCTIONS - COMMENTED OUT (no longer imported by any module)
These functions were part of the old extraction system and are no longer used:
- llm_generation_while_loop
- get_structured_generation_from_raw_generation_by_llm  
- get_structured_generation_from_raw_generation

They have been replaced by the new marker-based extraction system:
- llm_generation_with_extraction
- llm_generation_with_multiple_extractions
- extract_field

## Function:
#   llm inference with the prompt + guarantee to reply a structured generation accroding to the template (guarantee by the while loop)
#   gene_format_constraint: [id of structured gene to comply with the constraint, constraint (['Yes', 'No'], where the content in the id of structured gene should be inside the constraint)]
#   if_only_return_one_structured_gene_component: True or False; most of the time structured_gene will only have one component (eg, [[hyp, reasoning process]]). When it is True, this function will only return the first element of structured_gene. If it is set to true and structured_gene has more than one component, a warning will be raised
#   restructure_output_model_name: the model name used to extract structured generation if the original generation does not match the template. It is set in case some used model (model_name) is not powerful enough to follow the template, and in this case we can still extract the desired structured generation by using a more powerful model (restructure_output_model_name) to extract the structured generation from the original generation
def llm_generation_while_loop(prompt, model_name, client, if_structured_generation=False, template=None, gene_format_constraint=None, if_only_return_one_structured_gene_component=False, temperature=1.0, restructure_output_model_name=None, api_type=0):
    # assertions
    assert if_structured_generation in [True, False]
    if if_structured_generation:
        assert template is not None
    if restructure_output_model_name == None:
        restructure_output_model_name = model_name
    else:
        if restructure_output_model_name != model_name:
            print(f"Warning: restructure_output_model_name is set to {restructure_output_model_name}, which is different from model_name: {model_name}.")

    # while loop to make sure there will be one successful generation
    cnt_max_trials = 5
    generation = None
    for cur_trial in range(cnt_max_trials):
        try:
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            # print("generation: ", generation)
            # structured_gene
            if if_structured_generation:
                # structured_gene: [[title, reason], [title, reason], ...]
                # try with template matching first, if not work, try llm to formulate the generation according to the template; if not work again, then probably it is the problem of the original generation, then try llm_generation() again
                try:
                    # print("Using a template matching method to extract information from the LLM's generation")
                    structured_gene = get_structured_generation_from_raw_generation(generation, template=template)
                except:
                    # print("Information to be extracted by an LLM from the LLM's generation")
                    structured_gene = get_structured_generation_from_raw_generation_by_llm(generation, template=template, client=client, temperature=temperature, model_name=restructure_output_model_name, api_type=api_type)
                if gene_format_constraint != None:
                    assert len(gene_format_constraint) == 2, print("gene_format_constraint: ", gene_format_constraint)
                    # we use structured_gene[0] here since most of the time structured_gene will only have one component (eg, [[hyp, reasoning process]])
                    assert structured_gene[0][gene_format_constraint[0]].strip() in gene_format_constraint[1], print("structured_gene[0][gene_format_constraint[0]].strip(): {}; gene_format_constraint[1]: {}".format(structured_gene[0][gene_format_constraint[0]].strip(), gene_format_constraint[1]))
                # print("structured_gene: ", structured_gene)
            break
        except Exception as e:
            # if the format of feedback is wrong, try again in the while loop
            print("generation: ", generation)
            print("AssertionError: {}, try again..".format(repr(e)))
            if cur_trial == cnt_max_trials - 1:
                raise Exception("Failed to get generation after {} trials because of Error: {}.".format(cnt_max_trials, e))

    # structured_gene
    if if_structured_generation:
        if if_only_return_one_structured_gene_component:
            if len(structured_gene) > 1:
                print("Warning: structured_gene has more than one component: ", structured_gene)
            return structured_gene[0]
        else:
            return structured_gene
    else:
        return generation
    

# UNUSED BUT KEPT FOR BACKWARD COMPATIBILITY (used by llm_generation_while_loop which is also unused)
def get_structured_generation_from_raw_generation_by_llm(gene, template, client, temperature, model_name, api_type):
    assert isinstance(gene, str), print("type(gene): ", type(gene))
    # use .strip("#") to remove the '#' or "*" in the gene (the '#' or "*" is usually added by the LLM as a markdown format); used to match text (eg, title)
    gene = re.sub("[#*]", "", gene).strip()
    assert len(template) == 2, print("template: ", template)
    # In your answer, please only mention the words in the template when use it as a template. For example, if the template is ['Hypothesis:', 'Reasoning Process:'], then your answer should not contain 'Analysis of the Hypothesis:', since it also contain 'Hypothesis:'.
    # Whenever there are information in the passage related to the template, please restructure the information into the template format;
    prompt = "You are a helpful assistant.\nPlease help to organize the following passage into a structured format, following the template. When restructure the passage with the template, please try not to rephrase but to use the original information in the passage (to avoid information distortion). If the template is only about a subset of information in the passage, you can extract only that subset of information to fill the template. If there is no such information for the template in the passage, please still output the exact template first, and fill the content for the template as 'None'. \n\nThe passage is: \n" + gene + f"\n\nThe template is: \n{template[0]} \n{template[1]} \n. Now, please restructure the passage strictly with the template (literally strictly, e.g., the case style of the template should also remain the same when used to restructure the passage)."
    # print("prompt: ", prompt)
    
    # while loop to make sure there will be one successful generation
    max_trials = 10
    for cur_trial in range(max_trials):
        try:
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            # print("generation (in): ", generation)
            structured_gene = get_structured_generation_from_raw_generation(generation, template=template)
            # print("structured_gene (in): ", structured_gene)
            return structured_gene
        except Exception as e:
            if temperature < 2.0:
                temperature += 0.25
            # Q: do not change to more powerful model, since different users might have different model_name (even for the same model)
            # if temperature >= 0.7:
            #     model_name = "gpt-4o"
            # if the format of feedback is wrong, try again in the while loop
            print("generation (in): ", generation)
            print("template: ", template)
            print("Exception (in): {}, try again..".format(repr(e)))
            print(f"update temperature to {temperature} and use {model_name} for extraction in case new generation can be successful..")
    # print("structured_gene: ", structured_gene)
    raise Exception("Failed to restructure the passage with the template after {} trials.".format(max_trials))




# UNUSED BUT KEPT FOR BACKWARD COMPATIBILITY (used by unused llm_generation_while_loop functions)
# gene: (generated) text; '#' and '*' will be removed from gene, since they are assumed to be generated by LLM as markdown format --- this format can result in not exact match between the title extracted from generation and the groundtruth title in the benchmark
# template: ['Title:', 'Reason:']
# structured_gene: [[Title, Reason], ...]
def get_structured_generation_from_raw_generation(gene, template):
    # use .strip("#") to remove the '#' or "*" in the gene (the '#' or "*" is usually added by the LLM as a markdown format); used to match text (eg, title)
    gene = re.sub("[#*]", "", gene).strip()
    assert len(template) == 2, print("template: ", template)
    # # some times generation will capitalize the first letter of the template, so we use the lower case for both generation and template to match: not adopting it since it might influence chemistry terms (e.g., Fe2+ -> fe2+)
    # gene = gene.lower()
    # template = [item.lower() for item in template]
    # the template might not appear in the first sentence of the gene, get rid of noise sentences before the first template[0]
    if not gene.startswith(template[0]):
        gene_split = gene.split('\n')
        # if the gene is not starting with the title, the second paragraph in gene_split might be the title
        gene_split = [item for item in gene_split if item.strip() != ""]
        assert len(gene_split) >= 2, print("gene_split: ", gene_split)
        # iterate to find the first template[0] in gene_split
        for id_line, line in enumerate(gene_split):
            if gene_split[id_line].find(template[0]) > 0 and gene_split[id_line].find(template[0]) < 15:
                gene_split_split = gene_split[id_line].split(template[0])
                assert len(gene_split_split) == 2, print("gene_split_split: ", gene_split_split)
                gene_split[id_line] = template[0] + gene_split_split[1]
            if gene_split[id_line].startswith(template[0]):
                gene = '\n'.join(gene_split[id_line:])
                break
        # assert gene.startswith(template[0]), print("gene: ", gene)
        assert gene.startswith(template[0])
    # structured_gene: [[title, reason], [title, reason], ...]
    structured_gene = []
    gene_split = gene.split(template[0])
    # split to every title block, including one title and one reason
    for cur_gs in gene_split:
        # split the one title and one reason
        cur_gs = cur_gs.strip()
        if cur_gs == "":
            continue
        cur_gs_split = cur_gs.split(template[1])
        # deal with unexpected situations
        if len(cur_gs_split) > 2:
            # if there are more than one template[1] in cur_gs, we prefer the one with prefix as '\n' (since it matches more as the designed format)
            cur_gs_split = cur_gs.split('\n' + template[1])
            # by preferring the one with prefix as '\n' still do not work, so we simply concatenate the rest of the elements other than the first element
            if len(cur_gs_split) > 2:
                cur_gs_split = [cur_gs_split[0], '\n'.join(cur_gs_split[1:])]
            # in case none of the template[1] is with prefix as '\n'
            elif len(cur_gs_split) == 1:
                cur_gs_split = cur_gs.split(template[1])
                cur_gs_split = [cur_gs_split[0], '\n'.join(cur_gs_split[1:])]
        # assert len(cur_gs_split) == 2, print("cur_gs_split: ", cur_gs_split)
        assert len(cur_gs_split) == 2
        # strip every elements in cur_gs_split
        for i in range(len(cur_gs_split)):
            cur_gs_split[i] = cur_gs_split[i].strip().strip(";").strip()
        structured_gene.append(cur_gs_split)
    return structured_gene
"""  # END OF UNUSED FUNCTIONS


def extract_scores(cur_generation):
    """
    Extract evaluation scores using a unified approach that works for all model types.
    
    Returns: (scores, reasons, success) tuple where:
        - scores: list of 4 integer scores [validness, novelty, significance, specificity]
        - reasons: list of 4 strings with reasons for each score
        - success: boolean indicating if all scores were extracted
    """
    import re
    
    # Clean up reasoning model output
    cleaned_text = extract_reasoning_model_content(cur_generation)
    
    # Define the score types we're looking for
    score_types = ["Validness", "Novelty", "Significance", "Specificity"]
    scores = []
    reasons = []
    
    for score_type in score_types:
        # Try marker-based extraction first
        reason = extract_field(cleaned_text, f"{score_type} Reason", expected_type='text', strict_extraction=True)
        score = extract_field(cleaned_text, f"{score_type} Score", expected_type='number', strict_extraction=True)
        
        # If marker extraction failed, try simpler patterns
        if score is None:
            # Try patterns like "Validness score: 4" or "Validness: 4"
            patterns = [
                rf"{score_type}\s+score\s*:\s*(\d+)",
                rf"{score_type}\s*:\s*(\d+)",
                rf"score\s+for\s+{score_type}\s*:\s*(\d+)",
            ]
            
            for pattern in patterns:
                match = re.search(pattern, cleaned_text, re.IGNORECASE)
                if match:
                    score_str = match.group(1)
                    if score_str in ["1", "2", "3", "4", "5"]:
                        score = int(score_str)
                        break
        
        # If we still don't have a reason but have a score, try to extract reason
        if score is not None and not reason:
            # Try pattern like "Concise reason for X score: ..."
            reason_pattern = rf"(?:Concise\s+)?reason\s+for\s+{score_type}\s+score\s*:\s*([^\n]+)"
            match = re.search(reason_pattern, cleaned_text, re.IGNORECASE)
            if match:
                reason = match.group(1).strip()
        
        scores.append(score)
        reasons.append(reason or "")
    
    # Check if all scores were extracted and are valid
    valid_scores = all(score is not None and 1 <= score <= 5 for score in scores)
    
    if valid_scores:
        return scores, reasons, True
    else:
        print(f"Score extraction failed. Extracted scores: {scores}")
        return [], [], False


def pick_score(cur_generation):
    """
    Legacy function for backward compatibility. Redirects to extract_scores.
    
    We have 4 categories with scores and reasons.
    This function now uses the simplified extract_scores implementation.
    """
    # Simply redirect to the new extract_scores function
    return extract_scores(cur_generation)


def jaccard_similarity(str1, str2):
    """Calculate Jaccard similarity between two strings."""
    words1 = set(str1.split())
    words2 = set(str2.split())
    intersection = words1.intersection(words2)
    union = words1.union(words2)
    if len(union) == 0:
        return 0
    return len(intersection) / len(union)



# some titles are generated by LLM, which might have slight different from the exact title extracted from the markdown file
# groundtruth_titles: [title, ...], extracted from markdown file
# title: title generated by LLM 
def title_transform_to_exact_version_of_title_abstract_from_markdown(title, groundtruth_titles, if_print_warning=True):
    assert if_print_warning in [True, False]
    # groundtruth_titles:  [title, ...]
    similarity_collector = []
    for cur_item in groundtruth_titles:
        cur_similarity = jaccard_similarity(title.lower(), cur_item.lower()) 
        similarity_collector.append(cur_similarity)
    # get the most similar one
    max_similarity = max(similarity_collector)
    max_similarity_index = similarity_collector.index(max_similarity)
    matched_title = groundtruth_titles[max_similarity_index]
    if max_similarity < 0.3 and if_print_warning:
        print("max_similarity: {}; original title: {}; \nmatched title: {}\n".format(max_similarity, title, matched_title))
    return matched_title, max_similarity


# dict_title_2_abstract: a dict with groundtruth title as key, and abstract as value
# groundtruth_titles: [title, ...], extracted from markdown file
# title: title generated by LLM, that might not be exactly the same as the groundtruth title key in dict_title_2_abstract
## Output
# value: the abstract corresponding to the title
def get_item_from_dict_with_very_similar_but_not_exact_key(dict_title_2_abstract, title):
    groundtruth_titles = list(dict_title_2_abstract.keys())
    try:
        value = dict_title_2_abstract[title]
    except:
        title, similarity = title_transform_to_exact_version_of_title_abstract_from_markdown(title, groundtruth_titles)
        value = dict_title_2_abstract[title]
    return value


## Function:
#   generated title might be different from the exact title in the groundtruth title list, this function is to recover the generated title to the exact version of the title in the groundtruth title list
# groundtruth_titles: [title, ...]
# title: title generated by LLM
def recover_generated_title_to_exact_version_of_title(groundtruth_titles, title):
    title = title.strip().strip('"').strip()
    recovered_title, similarity = title_transform_to_exact_version_of_title_abstract_from_markdown(title, groundtruth_titles)
    return recovered_title


## Function:
#   whether an element is in a list with a similarity threshold (if th element has a similarity larger than the threshold with any element in the list, return True)
def if_element_in_list_with_similarity_threshold(list_elements, element, threshold=0.7):
    element = element.strip().strip('"').strip()

    for cur_element in list_elements:
        cur_element = cur_element.strip().strip('"').strip()
        if jaccard_similarity(element.lower(), cur_element.lower()) > threshold:
            return True
    return False


# # UNUSED
# def save_with_json(data, file_dir):
#     with open(file_dir, 'w') as f:
#         json.dump(data, f)



# # UNUSED: Function: transfer a list to set, while maintaining order
# def ordered_set(input_list):
#     set_list = []
#     for item in input_list:
#         if item not in set_list:
#             set_list.append(item)
#     return set_list


# -------------------------------------------------------------
# New extraction functions for marker-based extraction
# -------------------------------------------------------------

def extract_between_markers(source: str, label_regex: str):
    """Return text between '<label> starts' and '<label> ends'.

    Parameters
    ----------
    source : str
        The raw LLM response.
    label_regex : str
        A *REGEX* describing the label (e.g. 'Research\\s*question' or
        'Inspiration\\s+1'). It should NOT contain the 'starts'/'ends'
        keywords; they are added internally.

    Returns
    -------
    str | None
        The extracted content with compacted whitespace, or None if the
        pattern is not found.
    """
    # Remove markdown emphasis to simplify matching.
    plain = re.sub(r'[\*_]+', '', source)

    pattern = rf'{label_regex}\s*starts\s*:?\s*([\s\S]+?)\s*{label_regex}\s*ends'

    m = re.search(pattern, plain, flags=re.IGNORECASE | re.DOTALL)
    if not m:
        return None

    content = m.group(1).strip()
    return content if content else None


def extract_reasoning_model_content(text):
    """
    Extract the actual answer from reasoning models that use <think> or <answer> tags.
    Also removes thinking patterns for cleaner extraction.
    Returns the content after the thinking section.
    
    Compatible with various reasoning models including DeepSeek-R1, o1, etc.
    """
    import re
    
    # Keep original text in case we need to return it
    original_text = text
    
    # First, check if this is DeepSeek-R1 format with <answer> tags
    answer_match = re.search(r'<answer>(.*?)</answer>', text, re.DOTALL | re.IGNORECASE)
    if answer_match:
        # Found answer tags - this is DeepSeek-R1 format
        text = answer_match.group(1).strip()
    else:
        # Handle malformed responses where content comes after </think>
        after_think_match = re.search(r'</think>\s*\n(.+)', text, re.DOTALL | re.IGNORECASE)
        if after_think_match:
            # Extract everything after </think>
            text = after_think_match.group(1).strip()
        else:
            # Remove any <think> content (complete or partial)
            patterns_to_remove = [
                r'<think>.*?</think>',  # Complete think tags
                r'<think>.*$',  # Unclosed think tag at end
                r'^.*</think>',  # Everything up to and including </think>
                r'</think>?\s*$',  # Trailing closing tag (with optional typo)
                r'</think>',  # Any remaining closing tags
                r'<think>',  # Any remaining opening tags
            ]
            
            cleaned_text = text
            for pattern in patterns_to_remove:
                cleaned_text = re.sub(pattern, '', cleaned_text, flags=re.DOTALL | re.IGNORECASE)
            text = cleaned_text
    
    # Check if we have marked sections (e.g., "**Field starts:**")
    has_marked_sections = bool(re.search(r'\*\*\w+.*(?:starts|ends)\*\*', text))
    
    if not has_marked_sections:
        # Only apply cleanup if we don't have marked sections
        # Remove common thinking/explanation patterns at the end
        thinking_patterns = [
            r',\s*as\s+requested\.\s*$',
            r'\.\s*This\s+precisely.*$',
            r'\.\s*I\'ll\s+.*$',
            r'\.\s*Let\s+me\s+.*$',
            r'without\s+additional\s+commentary.*$',
        ]
        
        for pattern in thinking_patterns:
            text = re.sub(pattern, '', text, flags=re.IGNORECASE | re.MULTILINE)
    
    # Final cleanup
    text = text.strip()
    
    # Return cleaned text, or original if cleaning resulted in empty string
    return text if text else original_text


def extract_field(text, field_name, expected_type='text', strict_extraction=False):
    """Universal field extraction with type awareness.
    
    Args:
        text: The LLM response text
        field_name: The field to extract (e.g., "Hypothesis", "Answer", "Redundant")
        expected_type: 'text', 'bool'/'yes_no', 'number', etc.
        strict_extraction: If True, only use marker extraction (no fallbacks)
    
    Returns:
        Extracted value in appropriate type, or None if extraction fails
    """
    import re
    
    # Clean up reasoning model output (remove <think> tags)
    cleaned_text = extract_reasoning_model_content(text)
    
    # Try marker extraction first (works for both types of models after cleaning)
    result = extract_between_markers(cleaned_text, field_name)
    
    # If strict extraction and no marker found, return None
    if strict_extraction and not result:
        return None
    
    # Process based on expected type
    if expected_type in ['bool', 'yes_no', 'boolean']:
        if result:
            result_lower = result.lower().strip()
            if result_lower in ['yes', 'true', '1', 'correct', 'valid']:
                return True
            elif result_lower in ['no', 'false', '0', 'incorrect', 'invalid']:
                return False
        
        # Simple fallback: check start of cleaned text
        if not strict_extraction:
            text_lower = cleaned_text.lower().strip()[:100]
            if any(word in text_lower for word in ['yes', 'true', 'correct']):
                return True
            if any(word in text_lower for word in ['no', 'false', 'incorrect']):
                return False
        return None
    
    elif expected_type == 'number':
        if result:
            numbers = re.findall(r'\d+', result)
            if numbers:
                return int(numbers[0])
        
        # Simple fallback: look for number in cleaned text
        if not strict_extraction:
            numbers = re.findall(r'\b(\d+)\b', cleaned_text[:200])
            if numbers:
                return int(numbers[0])
        return None
    
    else:  # Default to text extraction
        if result:
            return result.strip()
        
        # Simple fallback: look for pattern "field_name: value"
        if not strict_extraction:
            escaped_field = re.escape(field_name)
            # Try simple colon pattern
            pattern = rf"{escaped_field}[:\s]+(.+?)(?:\n|$)"
            match = re.search(pattern, cleaned_text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                # Remove common markers
                value = re.sub(r'^\*+|\*+$', '', value).strip()
                value = re.sub(r'^["\']|["\']$', '', value).strip()
                return value
        
        return None


# Helper function for extracting multiple numbered field pairs
def llm_generation_with_multiple_extractions(prompt, model_name, client, repeating_field_pattern, max_items=10, temperature=1.0, api_type=0, max_retries=3):
    """
    Generate LLM response and extract multiple numbered field pairs.
    Useful for extracting lists like Title 1/Reason 1, Title 2/Reason 2, etc.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: The model to use
        client: The API client
        repeating_field_pattern: List of tuples defining the pattern that repeats for each numbered item
                                 [(field_name, expected_type), ...]
                                 e.g., [("Title", "text"), ("Reason", "text")] 
                                 will extract Title 1/Reason 1, Title 2/Reason 2, etc.
        max_items: Maximum number of items to try extracting (default 10)
        temperature: Temperature for generation
        api_type: API type (0=OpenAI, 1=Azure, 2=Google)
        max_retries: Maximum number of retries if extraction fails
    
    Returns:
        List of extracted item dictionaries, e.g., [{"Title": "...", "Reason": "..."}, ...]
    """
    for attempt in range(max_retries):
        try:
            # Generate response
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            
            # Extract numbered field pairs
            extracted_items = []
            for i in range(1, max_items + 1):
                item = {}
                all_fields_found = True
                
                for field_name, expected_type in repeating_field_pattern:
                    # Try to extract this numbered field
                    value = extract_field(generation, f"{field_name} {i}", expected_type=expected_type, strict_extraction=True)
                    if value is None:
                        all_fields_found = False
                        break
                    item[field_name] = value
                
                # If we found all fields for this item, add it
                if all_fields_found:
                    extracted_items.append(item)
                else:
                    # Stop looking for more items once we miss one
                    break
            
            # If we got at least one complete item, consider it successful
            if extracted_items:
                return extracted_items
            
            # If this was the last attempt, return empty list
            if attempt == max_retries - 1:
                print(f"Warning: No complete items could be extracted after {max_retries} attempts.")
                return []
            
            # Otherwise, retry with slightly higher temperature
            temperature = min(temperature + 0.1, 1.5)
            print(f"No items extracted (attempt {attempt + 1}/{max_retries}), retrying...")
            
        except Exception as e:
            print(f"Error during extraction attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                return []
    
    return []


# New function to replace llm_generation_while_loop with simpler extraction
def llm_generation_with_extraction(prompt, model_name, client, expected_fields=None, temperature=1.0, api_type=0, max_retries=15):
    """
    Generate LLM response and extract structured fields using marker-based extraction.
    Includes retry logic if extraction fails.
    
    Args:
        prompt: The prompt to send to the LLM
        model_name: The model to use
        client: The API client
        expected_fields: List of tuples [(field_name, expected_type), ...]
                        e.g., [("Title", "text"), ("Reason", "text")]
                        Supported types: "text", "number", "yes_no" (or "bool"/"boolean")
        temperature: Temperature for generation
        api_type: API type (0=OpenAI, 1=Azure, 2=Google)
        max_retries: Maximum number of retries if extraction fails (default 15)
    
    Returns:
        If expected_fields is provided: dict with extracted fields
        Otherwise: raw generation text
    """
    # If no expected fields, just return raw generation
    if not expected_fields:
        generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
        return generation
    
    # Validate expected_fields format
    if not isinstance(expected_fields, list):
        raise ValueError("expected_fields must be a list of tuples [(field_name, expected_type), ...]")
    for field_info in expected_fields:
        if not isinstance(field_info, tuple) or len(field_info) != 2:
            raise ValueError(f"Each field must be a tuple of (field_name, expected_type). Got: {field_info}")
        field_name, expected_type = field_info
        if expected_type not in ['text', 'number', 'yes_no', 'bool', 'boolean']:
            raise ValueError(f"Unsupported expected_type '{expected_type}' for field '{field_name}'. "
                           f"Supported types: 'text', 'number', 'yes_no' (or 'bool'/'boolean')")
    
    # Try extraction with retries
    for attempt in range(max_retries):
        try:
            # Generate response
            generation = llm_generation(prompt, model_name, client, temperature=temperature, api_type=api_type)
            
            # Extract fields
            extracted = {}
            all_fields_extracted = True
            
            for field_name, expected_type in expected_fields:
                value = extract_field(generation, field_name, expected_type=expected_type, strict_extraction=True)
                extracted[field_name] = value
                
                # Check if extraction failed (None or empty for text fields)
                if value is None or (expected_type == 'text' and value == ""):
                    all_fields_extracted = False
                    print(f"Warning: Failed to extract field '{field_name}' (attempt {attempt + 1}/{max_retries})")
            
            # If all required fields were extracted successfully, return
            if all_fields_extracted:
                return extracted
            
            # If this was the last attempt, return what we got
            if attempt == max_retries - 1:
                print(f"Warning: Some fields could not be extracted after {max_retries} attempts.")
                return extracted
            
            # Otherwise, retry with slightly higher temperature
            temperature = min(temperature + 0.1, 1.5)
            
        except Exception as e:
            print(f"Error during extraction attempt {attempt + 1}: {e}")
            if attempt == max_retries - 1:
                # On final attempt, return empty dict for all fields
                return {field_name: None for field_name, _ in expected_fields}
    
    # Should not reach here, but return empty dict as fallback
    return {field_name: None for field_name, _ in expected_fields}




