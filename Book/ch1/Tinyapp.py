import scipy as sp
import matplotlib.pyplot as plt

def error(f, x, y):
	return sp.sum((f(x)-y)**2)

data = sp.genfromtxt("data//web_traffic.tsv", delimiter="\t") 
#print data.shape
x = data[:,0]
y = data[:,1]
print sp.sum(sp.isnan(y))
print sp.sum(sp.isnan(x))
x = x[~sp.isnan(y)]#see that we remove those elements from x those are null in y
y = y[~sp.isnan(y)]

plt.scatter(x,y)
plt.title("Web traffic over the last month")
plt.xlabel("Time")
plt.xlabel("Hits/Hour")
plt.xticks([w*7*24 for w in range(10)],   ['week %i'%w for w in range(10)]) 
plt.autoscale(tight=True) 
plt.grid() 
#plt.show()

fp1, residuals, rank, sv, rcond = sp.polyfit(x, y, 1, full=True)
print("Model parameters: %s" %  fp1)
f1 = sp.poly1d(fp1) 
print(error(f1, x, y))
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting 
#plt.plot(fx, f1(fx), 'g-', linewidth = 2) 
#plt.legend(["d=%i" % f1.order], loc="upper left")
#plt.show()

f2p = sp.polyfit(x,y,2)
print f2p
f2 = sp.poly1d(f2p)
print(error(f2, x, y))
fx = sp.linspace(0,x[-1], 1000) # generate X-values for plotting 
#plt.plot(fx, f2(fx), 'r-', linewidth = 2) 
#plt.legend(["d=%i" % f1.order], loc="upper right")
#plt.show()

inflection = 3.5*7*24 # calculate the inflection point in hours 
xa = x[:inflection] # data before the inflection point 
ya = y[:inflection] 
xb = x[inflection:] # data after 
yb = y[inflection:]
fa = sp.poly1d(sp.polyfit(xa, ya, 1)) 
fb = sp.poly1d(sp.polyfit(xb, yb, 1))
fa_error = error(fa, xa, ya) 
fb_error = error(fb, xb, yb) 
print("Error inflection=%f" % ( fa_error + fb_error))


fx = sp.linspace(0,xa[-1], 1000) # generate X-values for plotting 
plt.plot(fx, fa(fx), 'r-', linewidth = 2) 

fx = sp.linspace(inflection,xb[-1], 1000) # generate X-values for plotting 
plt.plot(fx, fb(fx), 'r-', linewidth = 2) 

plt.show()
