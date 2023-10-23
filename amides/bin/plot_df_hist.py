#!/usr/bin/env python3
"""This script creates a histogram of decision function values of benign samples created by a misuse
classification model.
"""

import seaborn
import os
import pandas
import numpy as np
import matplotlib.pyplot as plt

from amides.persist import Dumper


valid_result_path = None


def main():
    dumper = Dumper(os.getcwd())
    valid_result = dumper.load_object(valid_result_path)

    df_values = valid_result.predict
    labels = valid_result.data.validation_data.labels
    del valid_result

    benign_label_idcs = np.where(labels == 0)
    benign_df_values = df_values[benign_label_idcs]

    del df_values
    del benign_label_idcs

    df = pandas.DataFrame({"Decision function value": benign_df_values})
    df_min = benign_df_values.min()
    df_max = benign_df_values.max()

    bin_width = (df_max - df_min) / 100

    _, (ax1, ax2) = plt.subplots(nrows=1, ncols=2)
    seaborn.histplot(
        data=df,
        y="Decision function value",
        bins=np.arange(
            df_min,
            df_max + bin_width,
            bin_width,
        ),
        ax=ax1,
    )

    seaborn.violinplot(data=df, y="Decision function value", ax=ax2)
    ax2.sharey(ax1)
    plt.show()


if __name__ == "__main__":
    main()
