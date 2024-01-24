# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
# %%
n_sigma = 6.0
n_part = int(1e5)

x  = np.zeros(n_part)
px = np.zeros(n_part)
y  = np.zeros(n_part)
py = np.zeros(n_part)
z = np.zeros(n_part)
sigma_d = 7.5e-2
dp = np.random.uniform(0.1*sigma_d,3.1*sigma_d,n_part)
# %%
def cmp_weights(df):
    r2 = df['x']**2 + df['px']**2 + df['y']**2 + df['py']**2
    w = np.exp(-r2/2.)
    #r2_l = df['z']**2 + df['dp']**2
    #w *=np.exp(r2_l/2.)
    w/=np.sum(w)
    return w


def generate_pseudoKV_xpyp(i):
  not_generated = True
  while not_generated:
    u = np.random.normal(size=4)
    r = np.sqrt(np.sum(u**2))
    u *= n_sigma/r
    v = np.random.normal(size=4)
    r = np.sqrt(np.sum(v**2))
    v *= n_sigma/r
    R2 = u[0]**2 + u[1]**2 + v[0]**2 + v[1]**2
    if R2 <= n_sigma**2:
        x[i]  = u[0]
        px[i] = u[1]
        y[i]  = v[0]
        py[i] = v[1]
        not_generated = False
  return 
# %%
list(map(generate_pseudoKV_xpyp, range(len(x))))

df = pd.DataFrame({'x':x , 'y': y, 'px': px, 'py': py, 'z': z, 'dp':dp})
#df.to_parquet("initial_distribution.parquet")
# %%
df['weights'] = cmp_weights(df)
# %%
fig, ax = plt.subplots(nrows=2, ncols=2)
plt.sca(ax[0,0])
plt.scatter(df['x'], df['y'], s=1)
plt.ylabel("y")
plt.xlabel("x")

plt.sca(ax[0,1])
plt.scatter(df['x'], df['px'], s=1)
plt.ylabel("px")
plt.xlabel("x")

plt.sca(ax[1,0])
plt.scatter(df['y'], df['py'], s=1)
plt.ylabel("py")
plt.xlabel("y")

plt.sca(ax[1,1])
plt.scatter(df['dp'],df['z'], s=1)
plt.xlabel("dp/p")
plt.ylabel("z")

fig.tight_layout()
fig.savefig("initial_distribution.png")
# %%
aperture = np.arange(0.0, 6.1, 0.1)
intensity = []
for current_aperture in aperture:
    df_copy = df[df['x']**2 + df['px']**2 <= (current_aperture)**2]
    current_intensity = df_copy['weights'].sum()
    intensity.append(current_intensity)
fig, ax = plt.subplots()
plt.plot(aperture, 1.-np.array(intensity), c='r', lw=4)
plt.plot(aperture, [np.exp(-x**2/2.0) for x in aperture], c='k', linestyle='--')
plt.xlabel('x (sigma)')
plt.ylabel("relative accumulated losses")
fig.savefig("scraping_x.png")


# %%
