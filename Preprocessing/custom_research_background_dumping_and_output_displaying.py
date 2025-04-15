import os, json, argparse


def research_background_to_json(research_background_file_path):
    # YOUR RESEARCH QUESTION HERE
    research_question = '''
    What physical mechanisms control the onset, propagation, and arrest of slow-slip events, and how do they differ from those of fast (seismic) ruptures?

    This is perhaps the most fundamental and widely investigated question. It seeks to understand why some segments of faults slip slowly while others fail catastrophically. Factors being explored include effective normal stress, pore fluid pressure, temperature, mineralogy, fault roughness, and the presence of transitional frictional behaviours. Experimental rock mechanics, field observations, and numerical modelling are all used to investigate whether slow and fast earthquakes are points along a continuum or governed by distinct processes. This question addresses the core of earthquake physics—if we understand what distinguishes slow and fast ruptures, we can better assess seismic hazard and possibly predict rupture behaviour under evolving tectonic conditions.

    In this context, two research directions are particularly promising. The first is advancing physics-based modeling to better capture the mechanisms governing slow-slip events and fault rupture behaviors. The second is exploring how the most advanced AI can be meaningfully integrated into the modeling workflow — not only for data analysis or detection, but also to assist in model development, parameter exploration, uncertainty quantification, and hypothesis generation. Compared to monitoring, which often relies on existing datasets and established methods, modeling and AI offer greater opportunities for methodological innovation and deeper understanding of earthquake processes.
    '''

    # YOUR BACKGROUND SURVEY HERE
    background_survey = '''
    Slow earthquakes and transient slip events
    Slow-slip events (SSEs), low-frequency earthquakes (LFEs), and tectonic tremor have redefined our understanding of fault slip and earthquake nucleation. First systematically recognized in the early 2000s in places like southwest Japan and Cascadia, these phenomena involve shear slip on faults that occurs over timescales of days to months, releasing seismic moment without the damaging high-frequency radiation typical of regular earthquakes. Over the past two decades, evidence has accumulated that such transient events are not anomalies but may be integral to the fault slip budget in many subduction zones, and crucially, they may interact with or even precede major earthquakes.
    In recent years (2022–2025), advances in geodetic and seismic monitoring have driven a wave of high-resolution observations of SSEs and tectonic tremor. Dense GNSS networks and continuous seismic arrays now allow us to resolve small-amplitude, long-duration deformation signals that were previously below detection thresholds. For example, deep slow-slip events with durations of 10–30 days and magnitudes equivalent to Mw ~6–7 have been regularly documented in Cascadia and Nankai, often accompanied by bursts of LFEs. In Japan, GNSS and borehole strain-meter data reveal quasi-periodic slow slip episodes at depths of 30–40 km, while in New Zealand’s Hikurangi margin, both deep and shallow SSEs have been observed, including some within reach of direct offshore instrumentation.
    The interpretive challenge lies in understanding what controls the occurrence of these events, how they evolve in time and space, and - perhaps most critically - how they relate to the nucleation of fast ruptures. Several major earthquakes in recent years (e.g., the 2011 Tohoku earthquake, the 2014 Iquique earthquake, the 2023 Türkiye-Syria earthquake) have exhibited precursory anomalies, such as localized tremor bursts, microseismic swarms, or transient geodetic signals in the months leading up to the mainshock. However, the physical connection between these transients and eventual failure remains elusive.
    Traditional detection methods for SSEs and tremor include cross-correlation of seismic waveforms, template matching for LFEs, and time series inversion of GNSS displacement fields. These approaches, while powerful, are often region-specific, labor-intensive, and limited in their ability to scale across different tectonic settings or uncover subtle signals. This is where machine learning (ML) and AI methods have begun to make an impact.
    Recent efforts have applied supervised and unsupervised learning to automate the detection of tremor and LFEs, extract transient deformation from noisy GNSS signals, and classify spatiotemporal patterns of fault slip. For example, convolutional neural networks have been used to detect tremor in continuous seismic records with significantly higher sensitivity than traditional techniques. Variational autoencoders and clustering algorithms have been employed to reveal structure in geodetic time series data, isolating slow-slip signatures from seasonal and anthropogenic noise. Moreover, generative models and sequence-to-sequence architectures (e.g., transformers, LSTMs) have been proposed to model the temporal evolution of fault slip and forecast transient behaviours.
    Despite promising results, key challenges remain. These include the sparsity and heterogeneity of labelled SSE datasets, regional variability in fault behaviours, and the need to integrate multi-modal data sources (e.g., seismic, geodetic, hydrogeological) to robustly interpret tectonic transients. Another open question is whether AI can do more than automate detection—can it help formulate or test physical hypotheses about the mechanics of slow slip? For example, can models learn to associate precursor tremor activity or aseismic deformation patterns with increased likelihood of a large earthquake?

    P.S.: While early works applied classical ML models (e.g., CNNs, VAEs, LSTMs) for detection, denoising, and pattern recognition in geophysical data, these approaches remain limited in capacity and generalization. Recent advances — such as foundation models, graph neural networks (GNNs), physics-informed learning, and multi-modal transformers — open new directions beyond detection, towards hypothesis generation, physical reasoning, and interpretable modeling of complex fault systems. Emerging techniques like simulation-based inference, neural operators, and differentiable physics further enable AI to assist in model development, parameter search, and scientific discovery. More imagination is needed if choose to incorporate ML models.
    '''


    # Save the research question and background survey to a JSON file
    with open(research_background_file_path, "w") as f:
        json.dump([research_question.strip(), background_survey.strip()], f, indent=4)
    print("Research background saved to", research_background_file_path)




def write_hypothesis_to_txt(eval_file_path, output_dir):
    # Load the JSON file
    with open(eval_file_path, "r") as f:
        data = json.load(f)

    research_question = list(data[0].keys())[0]

    with open(output_dir, "w") as f:
        for cur_id in range(len(data[0][research_question])):
            cur_hypothesis = data[0][research_question][cur_id][0]
            cur_score = data[0][research_question][cur_id][1]
            f.write("Hypothesis ID: " + str(cur_id) + "\n")
            f.write("Averaged Score: " + str(cur_score) + "; ")
            f.write("Scores: " + str(data[0][research_question][cur_id][2]) + "\n")
            f.write("Number of rounds: " + str(data[0][research_question][cur_id][4]) + "\n")
            f.write(cur_hypothesis + "\n")
            f.write("\n\n")
    # print("len(data):", len(data))





if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--io_type", type=int, default=1, help="0: dumping input to json for MOOSE-Chem to load; 1: displaying output to txt for human reading")
    parser.add_argument("--custom_research_background_path", type=str, default="./custom_research_background.json", help="the path to the research background file. The format is [research question, background survey], and saved in a json file. ")
    parser.add_argument("--evaluate_output_dir", type=str, default="./Checkpoints/evaluation_GPT4o-mini_updated_prompt_apr_14.json", help="the path to the output file of evaluate.py")
    parser.add_argument("--display_dir", type=str, default="./hypothesis.txt", help="the path to the output file for displaying the hypothesis to human readers")
    args = parser.parse_args()

    assert args.io_type in [0, 1], "args.io_type should be either 0 or 1"

    if args.io_type == 0:
        research_background_to_json(args.custom_research_background_path)
    elif args.io_type == 1:
        assert os.path.exists(args.evaluate_output_dir), "The evaluate output file does not exist."
        write_hypothesis_to_txt(args.evaluate_output_dir, args.display_dir)
    else:
        raise ValueError("args.io_type should be either 0 or 1")
    