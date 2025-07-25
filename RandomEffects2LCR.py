# RandomEffects2LCR.py
# 
import numpy as np
from numpy.polynomial.hermite import hermgauss
from scipy.stats import norm
from scipy.special import logsumexp
from scipy.optimize import minimize
from tqdm import tqdm
import pandas as pd

class RandomEffects2LCR:
    """
    Random effects 2-class latent class model (Qu, Tan & Kutner 1996) 
    + Supports 'constraint_groups_b' so that certain sets of tests share the same b(d,i).
    + Standard errors from numeric Hessian of final log-likelihood.
    """

    def __init__(
        self,
        T_obs,
        constraint_groups_b=None,   # e.g. [[0,1],[2],[3,4]] => means b(d,0) = b(d,1), b(d,2) alone, b(d,3)=b(d,4)
        n_classes=2,
        n_quadrature=20,
        max_iter=1000,
        tol=1e-6,
        init_param=None,  # (tau, a, b)
        seed=42,
        verbose=True,
        position=0
    ):
        # 1) store data
        T_obs = np.asarray(T_obs, dtype=int)
        patterns, counts = np.unique(T_obs, axis=0, return_counts=True)
        self.patterns = patterns
        self.counts   = counts
        self.n_patterns, self.n_tests = patterns.shape
        self.n_classes = n_classes
        self.n_quadrature = n_quadrature
        self.max_iter = max_iter
        self.tol = tol
        self.seed = seed
        self.init_param = init_param
        self.verbose = verbose
        self.position = position
        # 2) constraints on b => list of list => define groupings
        #    default => no constraints => each test has its own b => shape(2, J) free
        #    if user sets e.g. [[0,1],[2],[3,4]] => that means we have 3 "b group" => 
        #    b(d,0)= b(d,1) => 1 param for that group, b(d,2) alone => 1 param, b(d,3)= b(d,4)=> 1 param
        # => total for b(d,.) => 3 free param for each d => 2*d => 6
        if constraint_groups_b is None:
            # each test is its own group
            self.constraint_groups_b = [[i] for i in range(self.n_tests)]
        else:
            self.constraint_groups_b = constraint_groups_b

        # 3) placeholders
        self.tau = None
        self.a   = None  # shape=(2, n_tests)
        self.b   = None  # shape=(2, n_tests) but partially constrained
        self.log_likelihoods = []
        self.sens = None
        self.spec = None
        self.sens_se = None
        self.spec_se = None
        self.expected_frequencies = None
        self.cov_params_ = None   # store final param covariance from numeric Hessian
        self.cov_params_louis = None

    def gauss_hermite_quadrature(self):
        x_gh, w_gh = hermgauss(self.n_quadrature)
        t_j = x_gh* np.sqrt(2)
        w_j = w_gh/ np.sqrt(np.pi)
        return t_j, w_j

    def initialize_parameters(self):
        if self.init_param is not None:
            self.tau, self.a, self.b = self.init_param
        else:
            np.random.seed(self.seed)
            self.tau= np.array([0.5, 0.5])  # for 2-class
            self.a  = np.random.normal(0,1,size=(2,self.n_tests))
            self.b  = np.random.normal(0,1,size=(2,self.n_tests))
        
        if self.verbose:
            print("Initialize Parameters:")
            print(self._make_param_df())

    def fit(self):
        """
        main EM:
          E-step => posterior
          M-step => 
             1) update tau by eq(A3)
             2) update (a,b) under constraints => partial
        end => compute final standard errors from numeric Hessian of log-likelihood
        """
        self.initialize_parameters()
        t_j, w_j= self.gauss_hermite_quadrature()
        pbar= tqdm(total=self.max_iter, desc=f"2LCR-EM {self.position}", position=self.position, leave=True)
        for it in range(self.max_iter):
            tau_old= self.tau.copy()
            a_old= self.a.copy()
            b_old= self.b.copy()

            # E-step
            h_djk= self._E_step(t_j, w_j)
            # M-step
            z_djk= self.counts[:, None, None]* h_djk
            # eq(A3) => update tau
            K_d= np.sum(z_djk, axis=(0,2)) # shape=(2,)
            total_k= np.sum(K_d)
            if total_k<1e-15:
                total_k = 1e-15
                print("No valid patterns? total_k=0")
            tau0_new= K_d[0]/ (total_k)
            tau1_new= K_d[1]/ (total_k)
            self.tau= np.array([tau0_new, tau1_new])
            # update a,b
            e_info= {
                'z_djk': z_djk,
                't_j': t_j,
                'w_j': w_j
            }
            new_ab= self._M_step_ab(e_info)
            self._unpack_ab_constrained(new_ab) #####

            # check diff
            diff= (
                np.max(np.abs(self.tau- tau_old))+
                np.max(np.abs(self.a- a_old))+
                np.max(np.abs(self.b- b_old))
            )
            # compute log-likelihood
            ll_val= self._compute_full_loglike(t_j, w_j)
            self.log_likelihoods.append(ll_val)

            if it%50==0:
                pbar.update(50)
                if self.verbose:
                    pbar.set_description(f"EM {self.position}, ll={ll_val:.4f}: diff={diff:.3e}, tau={self.tau[-1]:.4f}")

            if diff< self.tol:
                pbar.set_description(f"Converged @iter={it}, diff={diff:.3e}, ll={ll_val:.6f}")
                pbar.n= self.max_iter
                pbar.refresh()
                break
        else:
            tqdm.write("EM did not converge in max_iter.")
        pbar.close()

        # final => freq
        log_p_y, _ = self._compute_posterior(t_j, w_j)
        p_y = np.exp(log_p_y)  # p(T_i | theta)
        self.expected_frequencies= np.maximum(p_y* np.sum(self.counts), 1e-15)

        # compute numeric Hessian => param covariance => store in self.cov_params_
        self.cov_params_ = self._compute_numeric_cov(t_j,w_j)
        self.cov_params_louis = self._compute_observed_info_via_louis_analytic(self._pack_all_params_constrained(), t_j, w_j)
        # final => sens, spec, SE
        self.compute_sens_spec_se(t_j, w_j)

        return (
            self.tau, self.a, self.b,
            self.expected_frequencies,
            self.sens,
            self.spec,
            self.log_likelihoods
        )

    def _E_step(self, t_j, w_j):
        """
        E-step => posterior h_djk shape=(n_patterns, 2, n_quadrature)
        """
        # p(d,r,k)= cdf(a[d,r]+ b[d,r]* t_j)
        a_ = self.a[:,:,None]
        b_ = self.b[:,:,None]
        p_ = norm.cdf(a_+ b_* t_j[None,None,:])
        p_ = np.clip(p_,1e-15,1-1e-15)

        # log-lik => shape(n_patterns,2,n_quadrature)
        # vector form:
        # Compute log-likelihood per Eq. 2.7: sum_r [y_ir log Phi + (1-y_ir) log (1-Phi)]
        log_lik= (
            self.patterns[:,None,:,None]* np.log(p_[None,:,:,:]) +
            (1- self.patterns[:,None,:,None])* np.log(1- p_[None,:,:,:])
        )
        log_lik= np.sum(log_lik, axis=2)
        # Add log(tau[d]) and log(w_j[k]) per Eq. 2.9
        logtau= np.log(self.tau)
        log_lik+= logtau[None,:,None]
        log_lik+= np.log(w_j)[None,None,:]

        max_ = np.max(log_lik, axis=(1,2), keepdims=True)
        log_lik-= max_
        denom= np.log(np.sum(np.exp(log_lik), axis=(1,2), keepdims=True))
        log_h= log_lik- denom
        return np.exp(log_h)
    
    def _compute_posterior(self, t_j, w_j):
        a_ = self.a[:,:,None]
        b_ = self.b[:,:,None]
        p_ = norm.cdf(a_ + b_ * t_j[None,None,:])
        p_ = np.clip(p_, 1e-15, 1 - 1e-15)
        log_lik = (
            self.patterns[:,None,:,None] * np.log(p_[None,:,:,:]) +
            (1 - self.patterns[:,None,:,None]) * np.log(1 - p_[None,:,:,:])
        )
        log_lik = np.sum(log_lik, axis=2)  # log p(T_i | d, u_k)
        log_prob = np.log(self.tau)[None,:,None] + log_lik + np.log(w_j)[None,None,:]  # log [p(T_i | d, u_k) * tau_d * w_k]
        log_p_y = logsumexp(log_prob, axis=(1,2))  # log p(T_i)
        log_h = log_prob - log_p_y[:, None, None]  # log p(D = d, U = u_k | T_i)
        h_ = np.exp(log_h)  # p(D = d, U = u_k | T_i)
        return log_p_y, h_
    
    def _compute_full_loglike(self, t_j, w_j):
        a_ = self.a[:, :, None]
        b_ = self.b[:, :, None]
        p_ = norm.cdf(a_ + b_ * t_j[None, None, :])
        p_ = np.clip(p_, 1e-15, 1 - 1e-15)
        log_g = (self.patterns[:, None, :, None] * np.log(p_[None, :, :, :]) +
                (1 - self.patterns[:, None, :, None]) * np.log(1 - p_[None, :, :, :]))
        log_g = np.sum(log_g, axis=2)
        log_prob = np.log(self.tau)[None, :, None] + log_g + np.log(w_j)[None, None, :]
        log_likelihood = np.sum(self.counts * logsumexp(log_prob, axis=(1, 2)))
        return log_likelihood

    # ========== M-step for (a,b) under constraint =============

    def _M_step_ab(self, e_info):
        """
        param => [ a(0, all), a(1, all), b-group(0, all groups), b-group(1, all groups)]
        because we have constraint_goups_b => each group in 0.. len(constraint_groups_b)-1
        => we define a(d, i) no constraint => i in [0.. J-1]
            b(d, group) => group in [0.. M-1], M= len(constraint_groups_b]
        """
        param0= self._pack_ab_constrained()
        def negQ_grad(x):
            f,g= self._Q_ab_function_and_grad_constrained(x, e_info)
            return -f, -g
        res= minimize(
            fun=lambda z: negQ_grad(z)[0],
            x0= param0,
            jac=lambda z: negQ_grad(z)[1],
            method='L-BFGS-B',
            options={'disp':False,'maxiter':500,'ftol':1e-9}
        )
        if not res.success:
            print("[_M_step_ab] partial =>", res.message)
        return res.x

    def _Q_ab_function_and_grad_constrained(self, param, e_info):
        """
        same as _Q_ab_function_and_grad, but with constraint that certain sets of b(d, i) are equal.
        We'll unpack param => a(0,:), a(1,:), b(0, groupCount), b(1, groupCount).
        Then fill self.b accordingly => b(d, i) = b(d, group_of(i)).
        Then do the same logic as before => compute Q_val, partial wrt these param.
        """
        self._unpack_ab_constrained(param)

        z_djk= e_info['z_djk']  # shape=(n_patterns,2,n_quadrature)
        t_j= e_info['t_j']

        # 1) build p_{d,r,k}
        a_ = self.a[:,:,None]
        b_ = self.b[:,:,None]
        p_ = norm.cdf(a_+ b_* t_j[None,None,:])
        p_ = np.clip(p_,1e-15,1-1e-15)
        logp= np.log(p_)
        log1p= np.log(1- p_)

        # 2) pattern part => sum_{r} [ pattern[i,r]* log p(d,r,k)+ ... ] => shape => (n_patterns,2,n_quadrature)
        part_r= np.zeros((self.n_patterns,2, self.n_quadrature))
        for d in range(2):
            for r in range(self.n_tests):
                pat_r= self.patterns[:, r]
                # 利用广播：将 pat_r 扩展为 (n_patterns, 1)
                part_r[:, d, :] += pat_r[:, None] * logp[d, r, :] + (1 - pat_r)[:, None] * log1p[d, r, :]
                # logp_rk= logp[d,r,:]
                # log1p_rk= log1p[d,r,:]
                # for kk in range(self.n_quadrature):
                #     part_r[:,d,kk]+= pat_r* logp_rk[kk] + (1- pat_r)* log1p_rk[kk]
        
        # 全向量化 代码不好懂 速度还要快一点
        # part_r = np.tensordot(self.patterns, logp, axes=([1], [1])) + np.tensordot(1 - self.patterns, log1p, axes=([1], [1]))

        # sum => Q_val => plus also add const from log(tau[d]), log(w_j[k]) if you want the full. But partial wrt a,b is enough.
        # We'll do the same approach as the simpler code:
        N_d= np.sum(z_djk, axis=(0,2))
        # ignoring log(tau) partial => no effect
        # ignoring w_j partial => no effect
        # so Q => sum_{i,d,k} z_djk[i,d,k]* part_r[i,d,k] + const
        pattern_part= np.sum(z_djk* part_r)
        # we define Q_val= pattern_part + constant
        # but let's add the constant to get the full Q for inspection => sum_{i,d,k} z_djk[i,d,k]* [ log(tau[d])+ log(w_j[k]) ]
        # though not needed for gradient wrt a,b
        # do a minimal approach => partial wrt a,b => we just store pattern_part
        # for clarity let's do the constant anyway:
        tau0, tau1= self.tau
        const_part1= N_d[0]* np.log(tau0)+ N_d[1]* np.log(tau1)
        sum_id= np.sum(z_djk, axis=(0,1))  # shape=(n_quadrature,)
        # we don't have direct w_j => let's do it from e_info ??? we do => e_info['w_j']
        from math import log
        w_ = e_info['w_j']
        const_part2= 0
        for k_ in range(len(w_)):
            const_part2+= sum_id[k_]* np.log(w_[k_])
        Q_val= const_part1+ const_part2+ pattern_part

        # 3) gradient wrt param => param= [ a(0,0..J-1), a(1,0..J-1), b(0,0..M-1), b(1,0..M-1)]
        # let's define an array for grad
        grad= np.zeros(len(param))

        # partial wrt a(d,r): 
        # same formula => factor(d,r,i,k)= pdf(...) * [ pattern[i,r]/ p_ - (1- pattern[i,r])/(1- p_) ]
        # then multiply z_djk[i,d,k], sum_{i,k}
        # partial wrt b(d, groupOf(r)) => sum_{ i,k} z_djk[i,d,k] * factor(d,r,i,k)* t_j[k],
        # but if r1, r2 in same group => b(d,r1)= b(d, r2). => partial wrt that group => we sum the partial from r1, r2.

        # precompute pdf => shape(2, J, n_quadrature)
        pdf_ = np.zeros((2, self.n_tests, len(t_j)))
        for d_i in range(2):
            for r_i in range(self.n_tests):
                pdf_[d_i,r_i,:]= norm.pdf(self.a[d_i,r_i]+ self.b[d_i,r_i]* t_j)

        # define indexing:
        J= self.n_tests
        M= len(self.constraint_groups_b)  # number of groups
        # param indices:
        #  a(d=0,r=0..J-1) => offset 0..(J-1)
        #  a(d=1,r=0..J-1) => offset J..(2J-1)
        #  b(d=0, group=0..M-1) => offset 2J..(2J+ M -1)
        #  b(d=1, group=0..M-1) => offset (2J+M)..(2J+2M -1)

        def idx_a(d, r):
            return d*J+ r
        def idx_b(d, group):
            return 2*J+ d*M+ group

        # # we need a mapping from test r => group index
        # test2group= {}
        # for gidx, grp in enumerate(self.constraint_groups_b):
        #     for rr in grp:
        #         test2group[rr]= gidx

        # partial wrt a(d_i,r_i)
        # for d_i in range(2):
        #     for r_i in range(J):
        #         # shape => (n_quadrature,)
        #         p__ = p_[d_i,r_i,:]
        #         pdf__= pdf_[d_i,r_i,:]
        #         pat__= self.patterns[:, r_i]
        #         factor_ = np.zeros((self.n_patterns, len(t_j)))
        #         for k_ in range(len(t_j)):
        #             tmp= pdf__[k_]* ( pat__/(p__[k_]) - (1- pat__)/(1- p__[k_]) )
        #             factor_[:, k_]= tmp
        #         grad_a_val= np.sum(z_djk[:, d_i, :]* factor_)
        #         grad[idx_a(d_i,r_i)] = grad_a_val

        for d_i in range(2):
            for r_i in range(J):
                # p_[d_i, r_i, :] 的形状为 (n_quadrature,)
                # pdf_[d_i, r_i, :] 的形状为 (n_quadrature,)
                # self.patterns[:, r_i] 的形状为 (n_patterns,)
                pat__ = self.patterns[:, r_i]  # shape: (n_patterns,)
                # 将 p 和 pdf 扩展为 (1, n_quadrature) 以便广播
                p_val = p_[d_i, r_i, :][None, :]        # shape: (1, n_quadrature)
                pdf_val = pdf_[d_i, r_i, :][None, :]      # shape: (1, n_quadrature)
                # 将 pat__ 转换为 (n_patterns, 1)
                pat_val = pat__[:, None]                 # shape: (n_patterns, 1)
                # 计算因子：shape (n_patterns, n_quadrature)
                factor_ = pdf_val * (pat_val / p_val - (1 - pat_val) / (1 - p_val))
                # 计算对 a(d_i, r_i) 的梯度贡献：对所有模式和积分节点求和
                grad_a_val = np.sum(z_djk[:, d_i, :] * factor_)
                grad[idx_a(d_i, r_i)] = grad_a_val
                
        # partial wrt b(d_i, group)
        # => we sum partial across all r in that group
        # => factor => the same formula but * t_j[k]
        # for d_i in range(2):
        #     for gidx in range(M):
        #         group_tests= self.constraint_groups_b[gidx]
        #         grad_b_val= 0.0
        #         for r_i in group_tests:
        #             p__= p_[d_i,r_i,:]
        #             pdf__= pdf_[d_i,r_i,:]
        #             pat__= self.patterns[:, r_i]
        #             factor_ = np.zeros((self.n_patterns, len(t_j)))
        #             for k_ in range(len(t_j)):
        #                 tmp= pdf__[k_]*( pat__/(p__[k_]) - (1- pat__)/(1- p__[k_]) )* t_j[k_]
        #                 factor_[:, k_]= tmp
        #             grad_b_val+= np.sum(z_djk[:, d_i, :]* factor_)
        #         grad[idx_b(d_i,gidx)] = grad_b_val

        for d_i in range(2):
            for gidx in range(M):
                group_tests = self.constraint_groups_b[gidx]  # 当前组中的测试索引，长度 R
                # p_vals: shape (R, K)
                p_vals = p_[d_i, group_tests, :]  
                # pdf_vals: shape (R, K)
                pdf_vals = pdf_[d_i, group_tests, :]
                # pat_vals: shape (n_patterns, R)
                pat_vals = self.patterns[:, group_tests]
                
                # 扩展维度：
                # p_vals: (R, K) -> (1, R, K)
                # pdf_vals: (R, K) -> (1, R, K)
                # pat_vals: (n_patterns, R) -> (n_patterns, R, 1)
                # t_j: (K,) -> (1, 1, K)
                p_vals_exp = p_vals[None, :, :]
                pdf_vals_exp = pdf_vals[None, :, :]
                pat_vals_exp = pat_vals[:, :, None]
                t_j_exp = t_j[None, None, :]
                
                # 计算向量化因子，形状 (n_patterns, R, K)
                factor = pdf_vals_exp * (pat_vals_exp / p_vals_exp - (1 - pat_vals_exp) / (1 - p_vals_exp)) * t_j_exp
                
                # z_djk[:, d_i, :] 形状为 (n_patterns, K)，扩展为 (n_patterns, 1, K)
                z_current = z_djk[:, d_i, :][:, None, :]
                
                # 计算梯度贡献：在 axis=(0,1) 求和得到标量
                grad_b_val = np.sum(z_current * factor)
                
                # 将结果存入梯度向量相应位置
                grad[idx_b(d_i, gidx)] = grad_b_val

        return Q_val, grad

    # ============ pack/unpack ab with constraint ===========

    def _pack_ab_constrained(self):
        """
        param= [ a(0,0..J-1), a(1,0..J-1),
                 b(0,0.. M-1 ), b(1,0..M-1 ) ]
        M= number of groups in constraint_groups_b
        for each group g => b(d,g) => used for all tests in that group
        """
        param_list= []
        # a(d=0)
        param_list.extend(self.a[0,:])
        # a(d=1)
        param_list.extend(self.a[1,:])
        # b(d=0) => each group 1 param
        M= len(self.constraint_groups_b)
        for gidx in range(M):
            # pick one test from that group to represent
            # e.g. test = group[0]
            # self.b[0, test], store
            test0= self.constraint_groups_b[gidx][0]
            param_list.append( self.b[0,test0] )
        # b(d=1)
        for gidx in range(M):
            test0= self.constraint_groups_b[gidx][0]
            param_list.append( self.b[1,test0] )

        return np.array(param_list,dtype=float)

    def _unpack_ab_constrained(self, param):
        """
        read param => shape= 2J + 2M
        fill self.a, self.b accordingly
        """
        J= self.n_tests
        M= len(self.constraint_groups_b)
        idx=0
        # a(0)
        a0= param[idx: idx+ J]; idx+= J
        # a(1)
        a1= param[idx: idx+ J]; idx+= J
        self.a[0,:]= a0
        self.a[1,:]= a1

        # b(0)
        b0_group= param[idx: idx+ M]; idx+= M
        # b(1)
        b1_group= param[idx: idx+ M]; idx+= M

        # now fill self.b
        for gidx in range(M):
            tests= self.constraint_groups_b[gidx]
            val0= b0_group[gidx]
            val1= b1_group[gidx]
            for r in tests:
                self.b[0,r]= val0
                self.b[1,r]= val1

    # ============ numeric Hessian => standard errors ===========

    def _compute_numeric_cov(self, t_j, w_j, eps=1e-5):
        param_vec = self._pack_all_params_constrained()
        p = len(param_vec)
        H = np.zeros((p, p))
        negloglike = lambda x: self._negloglike(x, t_j, w_j)
        for i in range(p):
            param_plus = param_vec.copy()
            param_minus = param_vec.copy()
            param_plus[i] += eps
            param_minus[i] -= eps
            H[i, i] = (negloglike(param_plus) - 2 * negloglike(param_vec) + negloglike(param_minus)) / (eps ** 2)
            for j in range(i + 1, p):
                param_pp = param_vec.copy()
                param_mm = param_vec.copy()
                param_pm = param_vec.copy()
                param_mp = param_vec.copy()
                param_pp[i] += eps; param_pp[j] += eps
                param_mm[i] -= eps; param_mm[j] -= eps
                param_pm[i] += eps; param_pm[j] -= eps
                param_mp[i] -= eps; param_mp[j] += eps
                H[i, j] = H[j, i] = (negloglike(param_pp) - negloglike(param_pm) - 
                                    negloglike(param_mp) + negloglike(param_mm)) / (4 * eps ** 2)
        try:
            cov_mat = np.linalg.inv(H)
        except np.linalg.LinAlgError:
            print("Warning: Observed information matrix is singular, using pseudo-inverse.")
            cov_mat = np.linalg.pinv(H)
        return cov_mat
    
    # ============ compute_observed_info_via_louis_analytic ===========

    def _compute_observed_info_via_louis_analytic(self, param_vec, t_j, w_j):
        """
        基于 Louis 公式，使用解析梯度计算观察信息矩阵：
        I(theta) = -E[Hessian_complete | T] - Cov(score_complete | T).

        参数：
        - param_vec: 参数向量 [eta, a0, a1, b0_groups, b1_groups]，其中 eta = logit(tau0)
        - t_j: 高斯-赫米特积分节点
        - w_j: 高斯-赫米特积分权重

        返回：
        - cov: 参数的协方差矩阵 (I_mat 的逆)
        """
        # 解包参数
        eta, a0, b0_groups, a1, b1_groups = self._unpack_all_params_constrained(param_vec)
        tau0 = 1 / (1 + np.exp(-eta))  # 计算 tau0
        tau1 = 1 - tau0                # tau1 = 1 - tau0
        sigma = 1.0                    # 标准化随机效应标准差
        n_params = len(param_vec)      # 参数总数

        # 初始化得分和 Hessian 累加器
        S = np.zeros((self.n_patterns, n_params))  # 条件得分矩阵
        H_sum = np.zeros((n_params, n_params))     # Hessian 期望累加

        # 计算后验概率和权重
        _, h_djk = self._compute_posterior(t_j, w_j)  # h_djk: (n_patterns, 2, n_quadrature)
        z_djk = self.counts[:, None, None] * h_djk    # 加权计数

        # 参数索引
        idx_eta = 0
        idx_a0 = np.arange(1, 1 + self.n_tests)
        idx_a1 = np.arange(1 + self.n_tests, 1 + 2 * self.n_tests)
        M = len(self.constraint_groups_b)
        idx_b0 = np.arange(1 + 2 * self.n_tests, 1 + 2 * self.n_tests + M)
        idx_b1 = np.arange(1 + 2 * self.n_tests + M, 1 + 2 * self.n_tests + 2 * M)

        # 数值保护参数
        eps_denom = 1e-15  # 防止除零
        max_bound = 1e8    # 限制中间结果绝对值
        nan_detected = False  # NaN 检测标志

        # 遍历每个模式和 GH 节点
        for i in range(self.n_patterns):
            T_i = self.patterns[i, :]  # 当前模式的响应向量
            for k in range(self.n_quadrature):
                u_k = t_j[k] * sigma  # 随机效应值
                g_d0 = np.zeros(n_params)  # D=0 的得分
                g_d1 = np.zeros(n_params)  # D=1 的得分
                H_d0 = np.zeros((n_params, n_params))  # D=0 的 Hessian
                H_d1 = np.zeros((n_params, n_params))  # D=1 的 Hessian

                # eta 的得分和 Hessian
                g_d0[idx_eta] = 1 - tau0  # D=0
                g_d1[idx_eta] = -tau0     # D=1
                H_d0[idx_eta, idx_eta] = -tau0 * (1 - tau0)
                H_d1[idx_eta, idx_eta] = -tau0 * (1 - tau0)

                # a 和 b 的得分和 Hessian
                for j in range(self.n_tests):
                    # 类别 D=0
                    z0 = a0[j] + self.b[0, j] * u_k  # 注意使用 self.b[0, j]
                    p_val0 = norm.cdf(z0)
                    p_val0 = np.clip(p_val0, eps_denom, 1 - eps_denom)
                    phi_val0 = norm.pdf(z0)
                    phi_val0 = max(phi_val0, eps_denom)  # 防止 phi_val0 过小
                    s0 = phi_val0 * (T_i[j] / p_val0 - (1 - T_i[j]) / (1 - p_val0))
                    g_d0[idx_a0[j]] = s0
                    term1 = -z0 * s0
                    term2 = -phi_val0**2 * (T_i[j] / p_val0**2 + (1 - T_i[j]) / (1 - p_val0)**2)
                    H_d0[idx_a0[j], idx_a0[j]] = np.clip(term1 + term2, -max_bound, max_bound)
                    # b0 的组参数
                    for gidx, grp in enumerate(self.constraint_groups_b):
                        if j in grp:
                            g_d0[idx_b0[gidx]] += s0 * u_k
                            H_d0[idx_b0[gidx], idx_b0[gidx]] += u_k**2 * (term1 + term2)
                            H_d0[idx_a0[j], idx_b0[gidx]] = H_d0[idx_b0[gidx], idx_a0[j]] = u_k * (term1 + term2)
                            break

                    # 类别 D=1
                    z1 = a1[j] + self.b[1, j] * u_k  # 注意使用 self.b[1, j]
                    p_val1 = norm.cdf(z1)
                    p_val1 = np.clip(p_val1, eps_denom, 1 - eps_denom)
                    phi_val1 = norm.pdf(z1)
                    phi_val1 = max(phi_val1, eps_denom)
                    s1 = phi_val1 * (T_i[j] / p_val1 - (1 - T_i[j]) / (1 - p_val1))
                    g_d1[idx_a1[j]] = s1
                    term1 = -z1 * s1
                    term2 = -phi_val1**2 * (T_i[j] / p_val1**2 + (1 - T_i[j]) / (1 - p_val1)**2)
                    H_d1[idx_a1[j], idx_a1[j]] = np.clip(term1 + term2, -max_bound, max_bound)
                    # b1 的组参数
                    for gidx, grp in enumerate(self.constraint_groups_b):
                        if j in grp:
                            g_d1[idx_b1[gidx]] += s1 * u_k
                            H_d1[idx_b1[gidx], idx_b1[gidx]] += u_k**2 * (term1 + term2)
                            H_d1[idx_a1[j], idx_b1[gidx]] = H_d1[idx_b1[gidx], idx_a1[j]] = u_k * (term1 + term2)
                            break

                    # NaN 调试
                    if np.any(np.isnan([s0, s1, p_val0, p_val1, phi_val0, phi_val1])):
                        print(f"NaN at pattern {i}, node {k}, test {j}: p0={p_val0}, p1={p_val1}")
                        nan_detected = True

                # 加权累加得分和 Hessian
                weight0 = z_djk[i, 0, k]
                weight1 = z_djk[i, 1, k]
                S[i, :] += weight0 * g_d0 + weight1 * g_d1
                H_sum += weight0 * H_d0 + weight1 * H_d1

        # Louis 公式计算观察信息矩阵
        I_mat = -H_sum - (S.T @ S) / self.n_patterns

        # 数值稳定性处理
        if np.any(np.isnan(I_mat)):
            print("NaN detected in I_mat")
            nan_detected = True
        # I_mat += np.eye(n_params) * 1e-4  # 对角扰动增强稳定性

        # 计算协方差矩阵
        try:
            cov = np.linalg.inv(I_mat)
        except np.linalg.LinAlgError:
            print("Warning: I_mat is singular, using pseudo-inverse.")
            cov = np.linalg.pinv(I_mat)

        # 调试信息
        if nan_detected:
            print("NaN detected during computation. Check data or parameters.")

        return cov

    def _negloglike(self, param, t_j, w_j):
        """
        param => all dof => we set it => compute - log L
        We'll do eq(2.8). 
        """
        self._unpack_all_params_constrained(param)
        ll= self._compute_full_loglike(t_j,w_j)
        return -ll

    def _pack_all_params_constrained(self):
        """
        param => [ logit(tau0 ), a(0,0..J-1), a(1,0..J-1),
                   b-groups(0?), b-groups(1?) ]
        """
        arr= []
        # logit(tau0)
        t0= np.clip(self.tau[0],1e-9,1-1e-9)
        arr.append( np.log(t0/(1- t0)) )
        # then a(0), a(1) => each shape=(J,)
        arr.extend(self.a[0,:])
        arr.extend(self.a[1,:])
        # then b => constraint form
        b_con= self._pack_b_constrained()
        arr.extend(b_con[0])  # b(d=0)
        arr.extend(b_con[1])  # b(d=1)
        return np.array(arr,dtype=float)

    def _unpack_all_params_constrained(self, param):
        """
        inverse => read param => shape= 1 + 2J + 2*gCount
        """
        idx=0
        logt0= param[idx]
        idx+=1
        t0= 1/(1+ np.exp(-logt0))
        self.tau= np.array([t0, 1- t0])

        J= self.n_tests
        a0= param[idx: idx+ J]; idx+= J
        a1= param[idx: idx+ J]; idx+= J
        self.a[0,:]= a0
        self.a[1,:]= a1
        M= len(self.constraint_groups_b)
        b0_group= param[idx: idx+ M]; idx+= M
        b1_group= param[idx: idx+ M]; idx+= M
        # fill self.b
        for g in range(M):
            for r in self.constraint_groups_b[g]:
                self.b[0,r]= b0_group[g]
                self.b[1,r]= b1_group[g]
        return logt0, a0, b0_group, a1, b1_group
    
    def _pack_b_constrained(self):
        """
        returns (arr_d0, arr_d1) each shape=(M,)
        M= len(constraint_groups_b)
        """
        M= len(self.constraint_groups_b)
        arr0= np.zeros(M)
        arr1= np.zeros(M)
        for gidx in range(M):
            r0= self.constraint_groups_b[gidx][0]
            arr0[gidx]= self.b[0, r0]
            arr1[gidx]= self.b[1, r0]
        return (arr0, arr1)

    # =========== standard API for final results ===========

    def compute_sens_spec_se(self, t_j, w_j):
        if self.cov_params_ is None:
            raise RuntimeError("Must call fit() first or no cov_params_ is computed.")
        
        # Unpack parameters
        # tau0, tau1, a0, b0, a1, b1 = self._unpack_all_params_constrained(self._pack_all_params_constrained())
        a0= self.a[0,:]
        a1= self.a[1,:]
        b0= self.b[0,:]
        b1= self.b[1,:]
        n_tests = self.n_tests
        covp = self.cov_params_
        n_params = len(covp)
        K = len(t_j)
        
        # Compute SE and SP
        se_j = np.zeros(n_tests)
        sp_j = np.zeros(n_tests)
        for j in range(n_tests):
            se_j[j] = np.sum(w_j * norm.cdf(a1[j] + b1[j] * t_j))
            sp_j[j] = np.sum(w_j * norm.cdf(-a0[j] - b0[j] * t_j))
        
        # Map tests to b-constraint groups
        test_to_group = {}
        for g, group in enumerate(self.constraint_groups_b):
            for test in group:
                test_to_group[test] = g
        
        # Parameter indices
        idx_logit_tau0 = 0
        idx_a0 = np.arange(1, 1 + n_tests)
        idx_a1 = np.arange(1 + n_tests, 1 + 2 * n_tests)
        M = len(self.constraint_groups_b)
        idx_b0 = np.arange(1 + 2 * n_tests, 1 + 2 * n_tests + M)
        idx_b1 = np.arange(1 + 2 * n_tests + M, 1 + 2 * n_tests + 2 * M)
        
        # Compute gradients and standard errors
        se_se_j = np.zeros(n_tests)
        sp_se_j = np.zeros(n_tests)
        for j in range(n_tests):
            # Gradient for SE_j
            grad_se = np.zeros(n_params)
            phi_se = norm.pdf(a1[j] + b1[j] * t_j)
            grad_se[idx_a1[j]] = np.sum(w_j * phi_se)
            g = test_to_group[j]
            grad_se[idx_b1[g]] = np.sum(w_j * t_j * phi_se)
            
            # Gradient for SP_j
            grad_sp = np.zeros(n_params)
            phi_sp = norm.pdf(a0[j] + b0[j] * t_j)
            grad_sp[idx_a0[j]] = -np.sum(w_j * phi_sp)
            grad_sp[idx_b0[g]] = -np.sum(w_j * t_j * phi_sp)
            
            # Variances and standard errors
            var_se = grad_se @ covp @ grad_se.T
            var_sp = grad_sp @ covp @ grad_sp.T
            se_se_j[j] = np.sqrt(np.maximum(var_se, 0))
            sp_se_j[j] = np.sqrt(np.maximum(var_sp, 0))

            self.sens = se_j
            self.spec = sp_j
            self.sens_se = se_se_j
            self.spec_se = sp_se_j
        
    def get_results(self, se_true=None, sp_true=None):
        # 使用 self.sens 和 self.spec 作为均值
        se_mean = np.array(self.sens)
        sp_mean = np.array(self.spec)
        se_sd = np.array(self.sens_se)
        sp_sd = np.array(self.spec_se)
        
        # 计算 95% 置信区间
        z_score = 1.96  # 95% 置信水平的 Z 分数
        se_ci_lower = np.maximum(se_mean - z_score * se_sd, 0)
        se_ci_upper = np.minimum(se_mean + z_score * se_sd, 1)
        sp_ci_lower = np.maximum(sp_mean - z_score * sp_sd, 0)
        sp_ci_upper = np.minimum(sp_mean + z_score * sp_sd, 1)
        
        # 构造结果 DataFrame
        df_res = pd.DataFrame({
            'Test': [f"Test {i+1}" for i in range(self.n_tests)],
            'Se_Estimate': se_mean,
            'Se_Median': se_mean,  # 暂时设为 NaN，可通过 Bootstrap 计算
            'Se_SD': se_sd,
            'Se_CI_lower': se_ci_lower,
            'Se_CI_upper': se_ci_upper,
            'Sp_Estimate': sp_mean,
            'Sp_Median': sp_mean,  # 暂时设为 NaN，可通过 Bootstrap 计算
            'Sp_SD': sp_sd,
            'Sp_CI_lower': sp_ci_lower,
            'Sp_CI_upper': sp_ci_upper,
            'Se_True': se_true if se_true is not None else np.nan,
            'Sp_True': sp_true if sp_true is not None else np.nan,
        })
        
        df_param = self._make_param_df()
        return df_res, df_param

    def _make_param_df(self):
        # shape => tau(2,), a(2,J), b(2,J)
        c= np.hstack(( self.tau.reshape(-1,1), self.a, self.b ))
        col= ["tau"]+ [f"a_{r+1}" for r in range(self.n_tests)]+ [f"b_{r+1}" for r in range(self.n_tests)]
        df= pd.DataFrame(c, columns=col)
        df.insert(0,"Class",[f"Class {i}" for i in range(self.n_classes)])
        return df

    def compute_deviance(self):
        obs= self.counts
        exp= self.expected_frequencies
        mask= (obs>0)
        obs_= obs[mask]
        exp_= exp[mask]
        dev= 2* np.sum(obs_* np.log(obs_/ exp_))
        return dev

    def compute_degrees_of_freedom(self):
        # param dof => 1 for tau(0), plus 2*J for a(d,*), plus 2*M for b(d, group)
        M= len(self.constraint_groups_b)
        n_param= 1+ 2*self.n_tests+ 2*M
        n_possible= 2** self.n_tests
        df= n_possible- n_param -1
        return df

    def compute_std_errors(self):
        if self.cov_params_ is None:
            raise RuntimeError("Must call fit() first or no cov_params_ is computed.")
        covp = self.cov_params_
        p = len(covp)
        
        # 提取对角元素并修复负值
        diag_covp = np.diag(covp)
        if np.any(diag_covp < 0):
            print("Warning: Negative variances detected in covariance matrix. Clamping to zero.")
            diag_covp = np.maximum(diag_covp, 0)  # 负值置为 0
        
        # 计算标准误差
        se_all = np.sqrt(diag_covp)

        # 参数顺序: [logit(tau0), a(0,0..J-1), a(1,0..J-1), b(0, group=0..M-1), b(1, group=0..M-1)]
        J = self.n_tests  # 测试项数量
        M = len(self.constraint_groups_b)  # 约束组数量
        
        # 提取各参数的标准误差
        idx = 0
        se_logit_tau0 = se_all[idx]  # logit(tau0) 的标准误差
        idx += 1
        se_a0 = se_all[idx: idx + J]  # a0 的标准误差，长度 J
        idx += J
        se_a1 = se_all[idx: idx + J]  # a1 的标准误差，长度 J
        idx += J
        se_b0_groups = se_all[idx: idx + M]  # b0 组标准误差，长度 M
        idx += M
        se_b1_groups = se_all[idx: idx + M]  # b1 组标准误差，长度 M
        
        # 计算 tau0 和 tau1 的标准误差
        tau0 = self.tau[0]  # 假设 tau0 已计算并存储
        se_tau0 = tau0 * (1 - tau0) * se_logit_tau0  # delta 方法
        se_tau1 = se_tau0  # 因为 tau1 = 1 - tau0，标准误差相同
        
        # 展开 b0_groups 和 b1_groups 的标准误差到长度 J
        se_b0 = np.zeros(J)  # 初始化 b0 标准误差数组
        se_b1 = np.zeros(J)  # 初始化 b1 标准误差数组
        for gidx, group in enumerate(self.constraint_groups_b):
            for test in group:
                se_b0[test] = se_b0_groups[gidx]  # 将组标准误差赋值给对应测试项
                se_b1[test] = se_b1_groups[gidx]
        
        # 返回结果字典
        return {
            'logit_tau0': se_logit_tau0,                # logit(tau0) 的标准误差
            'tau': np.array([se_tau0, se_tau1]),        # tau 的标准误差，shape=(2,)
            'a': np.array([se_a0, se_a1]),              # a 标准误差，shape=(2, J), 已展开
            'b': np.array([se_b0, se_b1])               # b 标准误差，shape=(2, J), 已展开
        }







