import logging
from typing import Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MultipleLocator

from symdesign import metrics


# logger = logging.getLogger(__name__)
# logger.setLevel(10)
mouse_dhfr = 'MVRPLNCIVAVSQNMGIGKNGDLPWPPLRNEFKYFQRMTTTSSVEGKQNLVIMGRKTWFSIPEKNRPLKDRINIVLSRELKEPPRGAHFLAKSLDDALRLIEQPELASKVDMVWIVGGSSVYQEAMNQPGHLRLFVTRIMQEFESDTFFPEIDLGKYKLLPEYPGVLSEVQEEKGIKYKFEVYEKKD'
gfp_emerald = 'MVSKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLTYGVQCFARYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHKVYITADKQKNGIKVNFKTRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITLGMDELYK'
human_tau40 = 'MAEPRQEFEVMEDHAGTYGLGDRKDQGGYTMHQDQEGDTDAGLKESPLQTPTEDGSEEPGSETSDAKSTPTAEDVTAPLVDEGAPGKQAAAQPHTEIPEGTTAEEAGIGDTPSLEDEAAGHVTQARMVSKSKDGTGSDDKKAKGADGKTKIATPRGAAPPGQKGQANATRIPAKTPPAPKTPPSSGEPPKSGDRSGYSSPGSPGTPGSRSRTPSLPTPPTREPKKVAVVRTPPKSPSSAKSRLQTAPVPMPDLKNVKSKIGSTENLKHQPGGGKVQIINKKLDLSNVQSKCGSKDNIKHVPGGGSVQIVYKPVDLSKVTSKCGSLGNIHHKPGGGQVEVKSEKLDFKDRVQSKIGSLDNITHVPGGGNKKIETHKLTFRENAKAKTDHGAEIVYKSPVVSGDTSPRHLSNVSSTGSIDMVDSPQLATLADEVSASLAKQGL'
"""human Tau-F 2N4R"""
benchmark_sequences = [
    mouse_dhfr,
    gfp_emerald,
    human_tau40
]


def compare_hydrophobicity_scale_to_standard(hydrophobicity_scale: dict[str, float],
                                             sequences: Sequence[Sequence] = None) -> float:
    """Test a new hydrophobicity scale again the standard scale from https://doi.org/10.1073/pnas.1617873114

    Returns:
        The computed value that is analogous for the given hydrophobicity_scale
    """
    if sequences is None:
        sequences = benchmark_sequences
        sequence_source = 'standard'
    else:
        sequence_source = 'provided'

    standard_collapse_indices = []
    new_collapse_indices = []
    for sequence in sequences:
        standard_collapse_index = metrics.hydrophobic_collapse_index(sequence, hydrophobicity='standard')
        standard_collapse_indices.append(standard_collapse_index)
        new_collapse_index = metrics.hydrophobic_collapse_index(sequence, hydrophobicity='custom',
                                                                custom=hydrophobicity_scale)
        new_collapse_indices.append(new_collapse_index)

    significance_threshold = metrics.collapse_thresholds['standard']
    equivalent_values = []
    for seq_idx, collapse in enumerate(standard_collapse_indices):
        collapse_bool = collapse > significance_threshold
        # Check for the collapse "transition" positions by comparing neighboring residues
        indices_around_transition_point = []
        for prior_idx, idx in enumerate(range(1, collapse_bool.shape[0])):
            # Condition is only True when 0 -> 1 transition occurs
            if collapse_bool[prior_idx] < collapse_bool[idx]:
                indices_around_transition_point.extend([prior_idx, idx])

        # Index the corresponding sequence from the new_collapse_indices
        equivalent_collapse_under_new_scale = new_collapse_indices[seq_idx][indices_around_transition_point]
        # equivalent_values.append(equivalent_collapse_under_new_scale.mean())
        equivalent_values.extend(equivalent_collapse_under_new_scale.tolist())

    print(f'For the {sequence_source} sequences, the equivalent hydrophobic collapse values are '
          f'{equivalent_values}')

    max_length = max([len(sequence) for sequence in sequences])
    new_significance_threshold = sum(equivalent_values) / len(equivalent_values)
    figure = False  # True  #
    # standard_collapse_indices_np = np.array(standard_collapse_indices)
    # new_collapse_indices_np = np.array(new_collapse_indices)
    if figure:
        # Set the base figure aspect ratio for all sequence designs
        figure_aspect_ratio = (max_length / 25., 20)  # 20 is arbitrary size to fit all information in figure
        fig = plt.figure(figsize=figure_aspect_ratio)
        collapse_ax = fig.subplots(1)
        for idx, indices in enumerate(standard_collapse_indices):
            collapse_ax.plot(indices, label=f'standard{idx}')
        for idx, indices in enumerate(new_collapse_indices):
            collapse_ax.plot(indices, label=f'new{idx}')
        # collapse_ax = collapse_graph_df.plot.line(ax=collapse_ax)
        collapse_ax.xaxis.set_major_locator(MultipleLocator(20))
        collapse_ax.xaxis.set_major_formatter('{x:.0f}')
        # For the minor ticks, use no labels; default NullFormatter.
        collapse_ax.xaxis.set_minor_locator(MultipleLocator(5))
        collapse_ax.set_xlim(0, max_length)
        collapse_ax.set_ylim(0, 1)
        # # CAN'T SET FacetGrid object for most matplotlib elements...
        # ax = graph_collapse.axes
        # ax = plt.gca()  # gca <- get current axis
        # labels = [fill(index, legend_fill_value) for index in collapse_graph_df.index]
        # collapse_ax.legend(labels, loc='lower left', bbox_to_anchor=(0., 1))
        # collapse_ax.legend(loc='lower left', bbox_to_anchor=(0., 1))
        # linestyles={'solid', 'dashed', 'dashdot', 'dotted'}
        # Plot horizontal significance
        collapse_ax.hlines([significance_threshold], 0, 1, transform=collapse_ax.get_yaxis_transform(),
                           label='Collapse Threshold', colors='#fc554f', linestyle='dotted')  # tomato
        collapse_ax.hlines([new_significance_threshold], 0, 1, transform=collapse_ax.get_yaxis_transform(),
                           label='New Collapse Threshold', colors='#cccccc', linestyle='dotted')  # grey
        collapse_ax.set_ylabel('Hydrophobic Collapse Index')
        plt.legend()
        plt.show()

    return new_significance_threshold


print('Starting calculation')
final_value = compare_hydrophobicity_scale_to_standard(metrics.hydrophobicity_scale['expanded'])
print(f'The average value of these new hydrophobicity scale is: {final_value}')

