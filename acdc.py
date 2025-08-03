"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
IMPLEMENTATION: Agglomerating and Competing Destination Choice
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

import numpy as np
from scipy import stats
from scipy.optimize import minimize

minval = 1E-12


def sigmoid(self, value):
    new_value = 1 / (1 + np.exp(-value))
    return new_value

''' ---------------------------------------------------------- '''
''' CLASS FOR ESTIMATION    '''
''' ---------------------------------------------------------- '''

'''
A spatially-aware discrete choice model designed to capture both agglomeration
effects (how nearby attractions enhance each other) and competition effects 
(how they substitute for one another).

Destination utility	- The perceived attractiveness of a location to the chooser.
Agglomeration -	Nearby destinations boost each other's attractiveness.
Competition	- Nearby destinations reduce each other's chance of being chosen.
Accessibility -	Ease of reaching a destination (e.g., travel time, cost).

More info:
1. Agglomeration (Cumulative Attraction):
•	Sites close to each other with complementary attributes (e.g., a beach with both 
camping and scenic views) increase the overall attractiveness of the area.
•	Inspired by Nelson (1958) and operationalized with accessibility measures that capture 
how nearby sites enhance a focal site

2. Competition (Substitution Effects):
•	Sites that share similar attributes compete, meaning one site's attractiveness reduces 
the appeal of nearby alternatives with the same features

3. Attribute-Specific Differentiation:
•	ACDC goes beyond using a single summary measure by allowing different accessibility variables
 for each attribute—such as campgrounds, playgrounds, or boat ramps—capturing nuanced substitution/agglomeration patterns

Assumption: locations are zones

Notation:
j,k - index for destination alternative 
m - index for attribute
n - index for trip
M - number of attributes
Z - number of attractiveness metrics
J[n] - choice set for trip n, J[n] is a list of destination
X[n][j][m] - value of attribute m for trip n and destination j
X[n] has shape (J[n] x M)  i.e., X is jagged
y[n] - destination chosen for trip n
V[n,j] - utility of selecting jth destination option for trip n

aggl[n][j][z] - agglomeration term for trip n, option j, metric k
aggl[n] is a list of arrays (J[n] x Z) 
'''

class ACDC:
# {
    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def __init__(self, **kwargs):
    # {
        self.descr = "ACDC"
        self.setup(**kwargs)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def setup(self, **kwargs):
    # {
        # Load dataframes
        self.trip_df = kwargs.get('trip_info')
        self.location_df = kwargs.get('location_info')
        self.person_df = kwargs.get('person_info')

        # Define #trips and #locns
        self.nb_trips = self.trip_df[self.trip_df['chosen'] == 1].shape[0]
        self.nb_locns = self.location_df.shape[0]  # The number of destinations

        # Define features of trips & people
        self.feature = self.trip_df.columns[6:].to_list() # Ignore first 6 columns "trip_id", "person_id", "origin_id", "dest_id", "alt", "chosen"
        self.feature += self.person_df.columns[1:].to_list() # Ignore first column "person_id"
        self.nb_features = len(self.feature)  # The total number

        # Define labels of location attractiveness
        self.attr = self.location_df.columns[4:].tolist()  # Ignore first 4 columns "locn_id", "locn_name", "x", "y"
        self.Z = len(self.attr) # Number of attractiveness measures
        self.aggl_labels = np.array([f"aggl_{self.attr[z]}" for z in range(self.Z)])
        self.comp_labels = np.array([f"comp_{self.attr[z]}" for z in range(self.Z)])
        self.labels = np.array(
            [f"{feature}" for feature in self.feature] +
            self.aggl_labels.tolist() +
            self.comp_labels.tolist()
        )

        # Merge location attributes
        self.df = self.trip_df.merge(self.location_df, left_on='dest_id', right_on='locn_id', how='left')
        self.df = self.df.drop(columns=['locn_id', 'locn_name'])

        # Merge person attributes
        self.df = self.df.merge(self.person_df, on='person_id', how='left')

        # Build trip_id to row mapping
        trip_ids = self.df['trip_id'].unique()
        trip_id_to_index = {tid: i for i, tid in enumerate(trip_ids)}

        # Fill X and y
        self.y = np.zeros(len(trip_ids), dtype=int)
        self.X = [] # Define X[n][j][m] where j is the destination in the choice set for trip n
        self.J = [] # Define J[n] the destination choice set
        for tid in trip_ids:
        # {
            i = trip_id_to_index[tid]
            df_n = self.df[self.df['trip_id'] == tid] # dataframe for trip n
            X_n = df_n[self.feature].to_numpy()
            self.X.append(X_n)
            self.J.append(df_n['dest_id'].tolist())

            chosen_row = df_n[df_n['chosen'] == 1]
            self.y[i] = chosen_row.index[0] - df_n.index[0]  # relative position
        # }

        self.get_agglomeration()
        self.get_competition()
        self.initialise()
    # }


    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    ''' Potential long format input
    def load_data_long(self):
    # {
        self.choice_df = kwargs.get('choice_data')  # long format: one row per (trip, destination)
        self.person_df = kwargs.get('person_info')  # optional if person data isn't already merged
        
        # Step 1: Build trip list
        self.trip_ids = self.choice_df['trip_id'].unique().tolist()
        self.nb_trips = len(self.trip_ids)
        
        # Step 2: Get attribute names (ignore first 6 known columns)
        self.attrs = self.choice_df.columns[6:].tolist()
        self.nb_attrs = len(self.attrs)
        
        # Step 3: Prepare X (jagged) and y (chosen alt index per trip)
        self.X = []
        self.y = np.zeros(self.nb_trips, dtype=int)
        
        trip_id_to_index = {tid: i for i, tid in enumerate(self.trip_ids)}
        
        for tid in self.trip_ids:
            i = trip_id_to_index[tid]
            
            df_n = self.choice_df[self.choice_df['trip_id'] == tid]
            
            # Build X[n] (J_n x M)
            X_n = np.zeros((df_n.shape[0], self.nb_attrs))
            for m, attr in enumerate(self.attrs):
                X_n[:, m] = df_n[attr].values
            self.X.append(X_n)
        
            # Find index of chosen alternative (must be exactly one per trip)
            chosen_idx = df_n['chosen'].values.argmax()
            self.y[i] = chosen_idx

    '''

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def initialise(self):
    # {
        self.nb_params = self.nb_features + 2 * self.Z
        self.params = np.zeros(self.nb_params)
        self.stderr = np.zeros(self.nb_params)
        self.signif_lb = np.zeros(self.nb_params)
        self.signif_ub = np.zeros(self.nb_params)
        self.pvalues = np.zeros(self.nb_params)
        self.zvalues = np.zeros(self.nb_params)
        self.loglike = None
        self.aic = None
        self.bic = None
    # }

    '''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def get_beta(self, params): return params[0:self.nb_features]

    # Weights are placed after the beta zone
    def get_gamma(self, params) -> np.ndarray: return params[-2:]

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # V[n][j] = sum(m, X[n][j][m] * beta[m]
    def compute_beta_X(self, beta):
    # {
        V = [] # List of arrays, one per trip
        for X_n in self.X:  # X_n shape is (J[n], M)
        # {
            V_n = X_n @ beta  # matrix-vector product. shape (J[n],)
            V.append(V_n)  # Add array of J[n] values
        # }
        return V
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Compute distance between location j and k
    def get_dist(self, j, k):
        delta = self.location_df.loc[j, ['x', 'y']] - self.location_df.loc[k, ['x', 'y']]
        return np.linalg.norm(delta)

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Get attractiveness measure z for location k
    def get_attr(self, k, z):
        return self.location_df.loc[k, self.attr[z]]

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Compute agglomeration term aggl[n][j][z]
    # Note: aggl is a list of arrays (J[n] x Z) for each trip n
    def get_agglomeration(self):
    # {
        self.aggl = []
        for n in range(self.nb_trips):
        # {
            nb_choices = len(self.J[n])
            aggl_n = np.zeros((nb_choices, self.Z), dtype=float)  # Create aggl[n] array
            for j in range(nb_choices):
            # {
                dest_j = self.J[n][j]   # id of jth alternative
                for k in range(nb_choices):
                # {
                    if k == j: continue   # Skip
                    dest_k = self.J[n][k]   # id of kth alternative
                    d_jk = self.get_dist(dest_j, dest_k)    # Compute distance between alternatives
                    if d_jk <= 0: continue  # Skip
                    for z in range(self.Z):
                        #aggl_n[j, z] += self.get_attr(dest_k, z) / np.exp(d_jk)  # Optional
                        aggl_n[j, z] += self.get_attr(dest_k, z) / d_jk
                # }
            # }
            self.aggl.append(aggl_n)
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Compute competition term comp[n][j][z]
    # Note: comp is a list of arrays (J[n] x Z) for each trip n
    def get_competition(self):
    # {
        self.comp = []  # Competition
        for n in range(self.nb_trips):
        # {
            nb_choices = len(self.J[n])
            comp_n = np.zeros((nb_choices, self.Z), dtype=float)
            for j in range(nb_choices): # choice index
            # {
                dest_j = self.J[n][j] # the jth destination
                for k in range(nb_choices): # choice index
                # {
                    if k == j: continue
                    dest_k = self.J[n][k]  # the kth destination
                    d_jk = self.get_dist(dest_j,dest_k)
                    if d_jk <= 0: continue
                    for z in range(self.Z):
                        comp_n[j][z] += self.get_attr(dest_k, z) * d_jk
                # }
            # }
            self.comp.append(comp_n)
        # }
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # For Bernadin Approach
    def get_dissimilarity(self):
    # {
        '''for j in range(self.nb_locns):
            for k in range(self.nb_locns):
                if k != j:
                    self.D[j][k] = 1
                    for g in G:
                        self.D[j][k] -= self.W[j][g] * self.W[k][g] / (self.B[j] * self.B[k])'''
        return 0
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # For Bernadin Approach
    def get_attraction(self):
    # {
        '''for j i n range(self.nb_locn):
            self.B[j] = 0
            for g in G:
                 self.B[j] += self.W[j][g] '''
        return 0
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # Extract the gamma weightings
    def unpack(self,params):
    # {
        gamma_agg = params[self.nb_features: self.nb_features + self.Z]
        gamma_comp = params[self.nb_features + self.Z: self.nb_features + 2 * self.Z]
        return gamma_agg, gamma_comp
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def compute_probability(self, params: np.ndarray)-> np.ndarray:
    # {
        beta = params[0:self.nb_features]   # M x 1
        gamma_agg, gamma_comp = self.unpack(params)
        utility = self.compute_beta_X(beta) # list of arrays, on per trip
        prob_list = []  # List
        for n in range(self.nb_trips):
        # {
            V_n = utility[n]    # shape: (J[n],)

            # Note: aggl and comp have shape (J[n] x Z)

            # Add agglomeration and competition terms
            V_n += self.aggl[n] @ gamma_agg     # shape: (J[n],)
            V_n += self.comp[n] @ gamma_comp    # shape: (J[n],)

            # Logit computation (numerical stability with max)
            V_n_stable = V_n - np.max(V_n)
            exp_V = np.exp(V_n_stable)
            prob_n = exp_V / np.sum(exp_V)

            prob_list.append(prob_n)
        # }
        return prob_list
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    # L(beta) = sum(n, sum(i, y[n,i] * log (P[n,i])))
    def get_loglike(self, params: np.ndarray)->float:
    # {
        self.probs = self.compute_probability(params)  # shape: (nb_trips,)
        indices = np.arange(self.nb_trips)
        chosen_probs = np.array([self.probs[n][self.y[n]] for n in range(self.nb_trips)])
        chosen_probs = np.clip(chosen_probs, 1e-20, 1.0)
        log_probs = np.log(chosen_probs)
        loglik = float(np.sum(log_probs))
        return loglik
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def evaluate(self, params: np.ndarray, minimize=True) -> float:
    # {
        self.loglike = self.get_loglike(params)
        score = self.loglike
        return -score if minimize else score
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def get_grad(self, params: np.ndarray, delta: np.ndarray):
        return self.compute_gradient_central(params, delta)

    ''' ---------------------------------------------------------- '''
    ''' Function                                                   '''
    ''' ---------------------------------------------------------- '''
    def get_loglike_gradient(self, params: np.ndarray, delta: np.ndarray):
        loglik = self.evaluate(params)
        grad = self.get_grad(params, delta)
        return (loglik, grad)

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_gradient_central(self, params: np.ndarray, delta: np.ndarray):
    # {
        gradient = np.array(np.zeros_like(params))  # create an array
        for i in range(len(params)):
        # {
            params_plus = params.copy()
            params_minus = params.copy()
            params_plus[i] += delta[i]
            params_minus[i] -= delta[i]
            case_1 = self.evaluate(params_plus)
            case_2 = self.evaluate(params_minus)
            gradient[i] = (case_1 - case_2) / (2.0 * delta[i])
        # }
        return gradient
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_bic(self, loglike):
        return np.log(self.nb_trips) * self.nb_params - 2.0 * loglike

    def get_aic(self, loglike):
         return 2.0 * self.nb_trips - 2.0 * loglike

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def fit(self, start=None):
    # {
        if start is None: start = np.zeros(self.nb_params)
        tol = 1e-10

        delta = np.ones(self.nb_params) * tol
        args = (delta,)  # tuple
        result = minimize(fun=self.get_loglike_gradient, x0=start, method='BFGS', args=args, tol=tol, jac=True)
        self.params = result.x  # Extract results
        self.post_process()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def get_hessian(self, eps=1e-6) -> np.array:
    # {
        N = self.nb_params  # Cardinality of hessian matrix
        hessian = np.zeros((N, N))  # Initialise hessian matrix
        delta = np.full(N, eps) # Create array of N values, each 'eps'
        params = np.array(np.copy(self.params))
        for i in range(N):  # i.e., for i = 0, 1, ..., N-1
        # {
            params[i] += eps
            df_plus = self.compute_gradient_central(params, delta)
            params[i] -= 2 * eps
            df_minus = self.compute_gradient_central(params, delta)
            hessian[i, :] = (df_plus - df_minus) / (2 * eps)
            params[i] += eps  # reset beta[i]
        # }
        return hessian

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_stderr(self, tol):
    # {
        hessian = self.get_hessian(tol)
        inverse = np.linalg.pinv(hessian)  # Conventional approach
        diag = np.diagonal(inverse)
        diag_copy = np.array(np.copy(diag))
        diag_copy[diag_copy < minval] = 0
        self.stderr = np.sqrt(diag_copy)
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_confidence_intervals(self):
        self.signif_lb = self.params - 1.96 * self.stderr  # i.e. signif_lb[m] = params[m] - 1.96 * stderr[m]
        self.signif_ub = self.params + 1.96 * self.stderr  # i.e.,signif_ub[m] = params[m] + 1.96 * stderr[m]

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_zvalues(self):
    # {
        for m in range(self.nb_params):
        # {
            if self.stderr[m] > minval:
                self.zvalues[m] = self.params[m] / self.stderr[m]
            else:
                self.zvalues[m] = np.nan
        # }
        self.zvalues = np.clip(self.zvalues, -np.inf, np.inf)  # Set limits
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def compute_pvalues(self):
    # {
        if self.nb_params < 100:
            self.pvalues = 2.0 * (1.0 - stats.t.cdf(np.abs(self.zvalues), df=self.nb_params))
        else:
            self.pvalues = 2.0 * (1.0 - stats.norm.cdf(np.abs(self.zvalues)))
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def post_process(self):
    # {
        self.loglike = self.evaluate(self.params, False)
        self.aic = self.get_aic(self.loglike)
        self.bic = self.get_bic(self.loglike)
        self.compute_stderr(1E-2) #1E-6)
        self.compute_zvalues()
        self.compute_pvalues()
        self.compute_confidence_intervals()
    # }

    ''' ---------------------------------------------------------- '''
    ''' Function.                                                  '''
    ''' ---------------------------------------------------------- '''
    def report(self):
    # {
        dp = 6
        fmt = f".{dp}f"
        np.set_printoptions(precision=dp, suppress=True)

        print("=" * 100)
        print(f"Method: ACDC")
        print(f"Log-Likelihood: {self.loglike:{fmt}}")
        print(f"AIC: {self.aic:{fmt}}")
        print(f"BIC: {self.bic:{fmt}}")

        # Print out table:
        print("{:>10} {:>12} {:>12} {:>12} {:>12} {:>12} {:>12}"
                .format("Coeff", "Estimate", "Std.Err.", "z-val", "p-val", "[0.025", "0.975]"))
        print("-" * 100)
        cond = "{:>10} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f} {:>12.6f}"
        for m in range(self.nb_params):
            formatted_str = cond.format(self.labels[m], self.params[m], self.stderr[m],
                        self.zvalues[m], self.pvalues[m], self.signif_lb[m], self.signif_ub[m])
            if self.pvalues[m] < 0.05: formatted_str += (" (*)")
            print(formatted_str)
        # }
        print("=" * 100)
    # }
# }

def main():
    """Main function to demonstrate the ACDC model."""
    import pandas as pd
    import os
    import sys
    # Example usage
    print("ACDC Model Example")
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))

# Change the working directory to the script's directory
    os.chdir(script_dir)
    # Note: You need to provide trip_info, location_info, and person_info DataFrames
    trip_info_df = pd.read_csv('trips.csv')  # DataFrame with trip information
    location_info_df = pd.read_csv('location.csv')  # DataFrame with location attributes
    person_info_df = pd.read_csv('person.csv')
    acdc_model = ACDC(trip_info=trip_info_df, location_info=location_info_df, person_info=person_info_df)
    acdc_model.fit()
    acdc_model.report()

if __name__ == "__main__":
    main()