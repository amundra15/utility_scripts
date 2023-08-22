import pandas as pd
import matplotlib.pyplot as plt

# Carpentries link for gapminder data
data_url = 'http://bit.ly/2cLzoxH'
#load gapminder data from url as pandas dataframe
gapminder = pd.read_csv(data_url)
print(gapminder.head(3))


error = [0, 2.99, 6.64, 10.63]
psnr = [32.59, 30.53, 29.97, 29.99]
lpips = [9.36, 14.41, 18.24, 21.85]


# create figure and axis objects with subplots()
fig,ax = plt.subplots()
# make a plot
ax.plot(error,
        psnr,
        color="tab:orange", 
        marker="o")
# set x-axis label
ax.set_xlabel("Average mesh error (mm)", fontsize = 14)
# set y-axis label
ax.set_ylabel(r'PSNR$\uparrow$',
              color="tab:orange",
              fontsize=14)


# twin object for two different y-axis on the sample plot
ax2=ax.twinx()
# make a plot with different y-axis using second axis object
ax2.plot(error, lpips,color="tab:blue",marker="o")
ax2.set_ylabel(r'LPIPS(x1000)$\downarrow$',color="tab:blue",fontsize=14)
plt.show()
# save the plot as a file
fig.savefig('two_different_y_axis_for_single_python_plot_with_twinx.jpg',
            format='jpeg',
            dpi=100,
            bbox_inches='tight')