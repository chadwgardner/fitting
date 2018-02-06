import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns
from my_stats_package import mean_squarred_error
sns.set()


def f(x):
    """function to model with a polynomial. Change this function to have anything modeled
    with polynomials and a quick, MSE analysis done!"""
    #Other suggested function:
    #return x * np.sin(x)
    return x**2 - 7*x + 11


def poly(x, coef):
    """Take an x-value and a list of polynomial coeffiecients (such as is returned
    by np.polyfit). Returns the corresponding value of the polynomial at that x.
    coef can be any length"""

    terms = []
    for i in range(len(coef)):
        terms.append(coef[i] * x ** ((len(coef)- 1 - i)))
    return sum(terms)


def make_new_data():
    """Specifically for this script. Creates a new, noisy dataset based on f(x),
    and saved as data.csv"""

    x = np.linspace(0, 10, 100)
    noise = np.random.normal(0, 1.5, size=len(x), )
    x_noisy = np.add(x, noise)

    # create matrix versions of these arrays
    #X = x[:, np.newaxis]
    #X_noisy = x_noisy[:, np.newaxis]
    #X_plot = x_plot[:, np.newaxis]

    y = f(x)
    y_noisy = f(x_noisy)

    data_stack = np.vstack((x, y_noisy)).T
    df = pd.DataFrame(data_stack, columns=['x', 'y_noisy'])
    df.to_csv('data.csv')

    return

#Read in the data
data = pd.read_csv('data.csv')
x_data = data['x']
y_data = data['y_noisy']

#Setting our test size to be used throughout
test_size = 0.4

#Split up the data into our training and test sets
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, test_size=test_size, random_state=42)

#Create plot to show all our data
plt.figure()
plt.scatter(x_data, y_data, marker='o', s=30, color='darkred', label='noisy data')
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.savefig('data.png', bbox_inches='tight')
plt.close()

#Create plot to show training and test data together
plt.figure(figsize=(10,5))
plt.subplot(1,2,1)
plt.scatter(X_train, y_train, marker='o', s=30, color='navy', label='training points')
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])

plt.subplot(1,2,2)
plt.scatter(X_test, y_test, marker='o', s=30, color='darkred', label='test points')
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.tight_layout()
plt.savefig('train_test.png', bbox_inches='tight')
plt.close()

#Getting our polynomial coeffients and residuals for each fit with degree 1, 2, and 10
#I ended up not using the residuals, but leaving here for future analysis.
#Notice that coefs is a list of lists, with each entry being the list of coeffients
#for our 1, 2, and 10 degree functions, respectively.
coefs = [[] for i in range(3)]
res = [[] for i in range(3)].
for count, degree in enumerate([1, 2, 10]):
    coefs[count], res[count], _, _, _ = np.polyfit(X_train, y_train, degree, full=True)

#Finding the points of the fits.
y_1d_fit_train = [poly(i, coefs[0]) for i in X_train]
y_2d_fit_train = [poly(i, coefs[1]) for i in X_train]
y_10d_fit_train = [poly(i, coefs[2]) for i in X_train]

y_1d_fit_test = [poly(i, coefs[0]) for i in X_test]
y_2d_fit_test = [poly(i, coefs[1]) for i in X_test]
y_10d_fit_test = [poly(i, coefs[2]) for i in X_test]

#Finding the MSE for each fit.
mse_1d_train = mean_squarred_error(y_train, y_1d_fit_train)
mse_2d_train = mean_squarred_error(y_train, y_2d_fit_train)
mse_10d_train = mean_squarred_error(y_train, y_10d_fit_train)

mse_1d_test = mean_squarred_error(y_test, y_1d_fit_test)
mse_2d_test = mean_squarred_error(y_test, y_2d_fit_test)
mse_10d_test = mean_squarred_error(y_test, y_10d_fit_test)

#Quick report to check the MSE for any oddities
report = pd.DataFrame(columns = ['Train', 'Test'], index=['1d', '2d', '10d'])
report.loc['1d'] = [mse_1d_train, mse_1d_test]
report.loc['2d'] = [mse_2d_train, mse_2d_test]
report.loc['10d'] = [mse_10d_train, mse_10d_test]
print(report)

#Textbox properties for plot labels
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)

#make x axis
x_plot = np.linspace(0, 10, 100)

#Make 1D Plots
plt.figure(figsize = (11,5))
plt.subplot(1,2,1)
plt.suptitle('First Degree Polynomial Fit')
plt.scatter(X_train, y_train, color='navy', s=30, marker='o', label="training points", alpha=0.4)
plt.plot(x_plot, poly(x_plot, coefs[0]), label='1D fit')
plt.margins(0.02)
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.title('1-Degree Training Data')
plt.text(5, 70, 'MSE: ' + str(np.around(mse_1d_train,decimals=2)), fontsize=14,
    verticalalignment='top', bbox=props)

plt.subplot(1,2,2)
plt.plot(x_plot, f(x_plot), color='darkred', linewidth=2, label="target", alpha=0.7)
plt.scatter(X_test, y_test, color='maroon', s=30, marker='o', label='test point', alpha=0.4)
plt.plot(x_plot, poly(x_plot, coefs[0]), label='1D fit: ' + str(np.around(mse_1d_test,decimals=2)))
plt.margins(0.02)
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.title('1-Degree Test Data')
plt.text(5, 70, 'MSE: ' + str(np.around(mse_1d_test,decimals=2)), fontsize=14,
    verticalalignment='top', bbox=props)

plt.savefig('1d.png', bbox_inches='tight')

#Make 2D Plots
plt.figure(figsize = (11,5))
plt.subplot(1,2,1)
plt.suptitle('2nd Degree Polynomial Fit')
plt.scatter(X_train, y_train, color='navy', s=30, marker='o', label="training points", alpha=0.4)
plt.plot(x_plot, poly(x_plot, coefs[1]), label ='2D fit')
plt.margins(0.02)
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.title('2-Degree Training Data')
plt.text(5, 70, 'MSE: ' + str(np.around(mse_2d_train,decimals=2)), fontsize=14,
    verticalalignment='top', bbox=props)

plt.subplot(1,2,2)
plt.plot(x_plot, f(x_plot), color='darkred', linewidth=2, label="target", alpha=0.7)
plt.scatter(X_test, y_test, color='maroon', s=30, marker='o', label='test point', alpha=0.4)
plt.plot(x_plot, poly(x_plot, coefs[1]), label ='2D fit')
plt.margins(0.02)
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.title('2-Degree Test Data')
plt.text(5, 70, 'MSE: ' + str(np.around(mse_2d_test,decimals=2)), fontsize=14,
    verticalalignment='top', bbox=props)

plt.savefig('2d.png', bbox_inches='tight')

#Make 10D Plots
plt.figure(figsize = (11,5))
plt.subplot(1,2,1)
plt.suptitle('10th Degree Polynomial Fit')
plt.scatter(X_train, y_train, color='navy', s=30, marker='o', label="training points", alpha=0.4)
plt.plot(x_plot, poly(x_plot, coefs[2]), label='10D fit')
plt.margins(0.02)
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.title('10-Degree Training Data')
plt.text(5, 70, 'MSE: ' + str(np.around(mse_10d_train,decimals=2)), fontsize=14,
    verticalalignment='top', bbox=props)

plt.subplot(1,2,2)
plt.plot(x_plot, f(x_plot), color='darkred', linewidth=2, label="target", alpha=0.7)
plt.scatter(X_test, y_test, color='maroon', s=30, marker='o', label='test point', alpha=0.4)
plt.plot(x_plot, poly(x_plot, coefs[2]), label='10D fit')
plt.margins(0.02)
plt.legend(loc='upper left')
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
plt.title('10-Degree Test Data')
plt.text(5, 70, 'MSE: ' + str(np.around(mse_10d_test,decimals=2)), fontsize=14,
    verticalalignment='top', bbox=props)

plt.savefig('10d.png', bbox_inches='tight')
plt.close('all')


#Bootstraping: Now running the same fitting anlysis repeated to demonstrate bootstrapping
#and to eventually show bias-variance tradeoff.

#Number of bootstraps
runs = 100

#Setting up the plot
plt.figure(figsize=(18,6))
plt.subplot(1,3,1)
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
one_d_coefs = []
one_d_mses = []

for i in range(runs):
    #Make bootstrap rep and plot the resulting polynomial - 1D
    bs_X_train, bs_X_test, bs_y_train, bs_y_test = train_test_split(x_data, y_data, test_size=test_size)
    one_d_coefs.append(np.polyfit(bs_X_train, bs_y_train, 1))
    plt.plot(x_plot, poly(x_plot, one_d_coefs[i]), color='blue', alpha=0.03)

    #Find MSE for the bootstrap
    bs_y_1d_fit_test = [poly(j, one_d_coefs[i]) for j in bs_X_test]
    one_d_mses.append(mean_squarred_error(bs_y_test, bs_y_1d_fit_test))

    #Plot mean of the runs
    if i == runs - 1:
        mean_one_d_coefs = []
        for k in range(2):
            mean_one_d_coefs.append(np.mean([j[k] for j in one_d_coefs]))
        plt.plot(x_plot, poly(x_plot, mean_one_d_coefs), color='darkblue')
        plt.title('1st Degree Fit')
    else:
        pass

#2D plot
plt.subplot(1,3,2)
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
two_d_coefs = []
two_d_mses = []

for i in range(runs):
    #Make bootstrap rep and plot the resulting polynomial - 2D
    bs_X_train, bs_X_test, bs_y_train, bs_y_test = train_test_split(x_data, y_data, test_size=test_size)
    two_d_coefs.append(np.polyfit(bs_X_train, bs_y_train, 2))
    plt.plot(x_plot, poly(x_plot, two_d_coefs[i]), color='b', alpha=0.03)

    #Find MSE for the bootstrap
    bs_y_2d_fit_test = [poly(j, two_d_coefs[i]) for j in bs_X_test]
    two_d_mses.append(mean_squarred_error(bs_y_test, bs_y_2d_fit_test))

    #Plot the mean of the runs
    if i == runs - 1:
        mean_two_d_coefs = []
        for k in np.arange(3):
            mean_two_d_coefs.append(np.mean([j[k] for j in two_d_coefs]))

        plt.plot(x_plot, poly(x_plot, mean_two_d_coefs), color='darkblue')
        plt.title('2nd Degree Fit')
    else:
        pass

#10D plot
plt.subplot(1,3,3)
plt.ylim(-5,75)
plt.xticks([])
plt.yticks([])
ten_d_coefs = []
ten_d_mses = []
for i in range(runs):
    #Make bootstrap rep and plot the resulting polynomial - 10D
    bs_X_train, bs_X_test, bs_y_train, bs_y_test = train_test_split(x_data, y_data, test_size=test_size)
    ten_d_coefs.append(np.polyfit(bs_X_train, bs_y_train, 10))
    plt.plot(x_plot, poly(x_plot, ten_d_coefs[i]), color='blue', alpha=0.03)

    #Find MSE for the bootstraps
    bs_y_10d_fit_test = [poly(j, ten_d_coefs[i]) for j in bs_X_test]
    ten_d_mses.append(mean_squarred_error(bs_y_test, bs_y_10d_fit_test))

    if i == runs - 1:
        mean_ten_d_coefs = []
        for k in np.arange(11):
            mean_ten_d_coefs.append(np.mean([j[k] for j in ten_d_coefs]))

        plt.plot(x_plot, poly(x_plot, mean_ten_d_coefs), color='darkblue')
        plt.title('10 Degree Fit')
    else:
        pass

plt.tight_layout()
plt.savefig('bootstrapped.png', bbox_inches='tight')
plt.clf()
plt.close()

# Plot MSE histograms

data = np.array([one_d_mses, two_d_mses, ten_d_mses])
data = np.transpose(data)
mses = pd.DataFrame(data, columns=['one_d', 'two_d', 'ten_d'])
mses = mses[mses['ten_d'] < 500]
sns.boxplot(data=mses)
plt.xlabel('Degree of Polynomial Model')
plt.ylabel('Mean Squared Error')
plt.legend(loc='best')
plt.savefig('boxplots_mse.png', bbox_inches='tight')
plt.close()

plt.figure()
plt.hist(mses['two_d'], label='Degree 2', alpha=0.8, bins=8)
plt.hist(mses['one_d'], label='Degree 1', alpha=0.8, bins=10)
plt.hist(mses['ten_d'], label='Degreee 10', alpha=0.8, bins=20)
plt.legend(loc='best')
plt.xlabel('Mean Squared Error')
plt.ylabel('Count')
plt.savefig('mse_hists.png', bbox_inches='tight')
plt.close()


#Bias-variance trade off exploration

#Making x-axis and setting up lists
powers = np.linspace(0, 10, 11)
plt.figure()
mses_train_ = []
mses_test_ = []
mses_train_means = []
mses_test_means = []

for i in range(runs):
    #for each run, split up the data
    var_X_test, var_X_train, var_y_test, var_y_train = train_test_split(x_data, y_data, test_size=test_size)
    mses_train = []
    mses_test = []

    #For every power in powers, make a polynomial fit to the current run's train and test data
    #Calculate the MSE
    #Plot the trend
    for p in powers:
        coeffiecients = np.polyfit(var_X_train, var_y_train, p)
        train = [poly(i, coeffiecients) for i in var_X_train]
        test = [poly(i, coeffiecients) for i in var_X_test]
        mses_train.append(mean_squarred_error(var_y_train, train))
        mses_test.append(mean_squarred_error(var_y_test, test))


    plt.plot(powers, mses_train, alpha=0.05, color='blue')
    plt.plot(powers, mses_test, alpha=0.05, color='red')
    mses_train_.append(mses_train)
    mses_test_.append(mses_test)

    #At the end, plot the mean of the runs
    if i == runs - 1:
        for k in range(len(powers)):
            mses_train_means.append(np.mean([j[k] for j in mses_train_]))
            mses_test_means.append(np.mean([j[k] for j in mses_test_]))
        plt.plot(powers, mses_train_means, label = 'Mean Training Fit', color='darkblue')
        plt.plot(powers, mses_test_means, label = 'Mean Testing Fit', color='darkred')
    else:
        pass

#Final plot attributes
plt.ylim(0,400)
plt.ylabel('Mean Squared Error')
plt.xlabel('<--High Bias/Low Variance | Polynomial Degree | Low Bias/High Variance -->')
plt.legend(loc='best')
plt.savefig('bias_variance.png', bbox_inches='tight')

#You're done!
