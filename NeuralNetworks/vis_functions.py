import matplotlib.pyplot as plt


def create_scatter_plot(dataset_name, v_act, l_v_act, p_act, l_act):
    plt.scatter(v_act, l_v_act, color='blue', label='Validation')
    plt.scatter(p_act, l_act, color='red', label='Training')
    plt.xlabel('Predicted')
    plt.ylabel('Observed')
    plt.title(dataset_name)
    plt.legend()
    plt.show()
