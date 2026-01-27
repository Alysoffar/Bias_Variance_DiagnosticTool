import matplotlib.pyplot as plt

def plot_learning_curve(sizes, train_errors, val_errors, title="Learning Curve" , out_path= None):
    plt.figure()

    plt.plot(sizes, train_errors, label="Training Error", marker='o')
    plt.plot(sizes, val_errors, label="Validation Error", marker='o')

    plt.xlabel("Training Set Size")
    plt.ylabel("Error")

    plt.title(title)
    plt.legend()
    plt.grid()

    if out_path:
        plt.savefig(out_path,bbox_inches="tight")
    else:
        plt.show()

        
    plt.close()
