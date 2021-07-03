误差由两部分组成，一部分来自于后一个单元反向传播记为$\mathrm{d}\bar{a}^{\left\langle t\right\rangle}$和$\mathrm{d}\bar{c}^{\left\langle t\right\rangle}$， 一部分来自于本单元损失函数的产生的误差$\mathrm{d}\hat{a}^{\left\langle t\right\rangle}$和$\mathrm{d}\hat{c}^{\left\langle t\right\rangle}$

因此有：
$$
\mathrm{d}{a}^{\left\langle t\right\rangle}=\mathrm{d}\bar{a}^{\left\langle t\right\rangle}+\mathrm{d}\hat{a}^{\left\langle t\right\rangle}
$$
$\mathrm{d}{c}^{\left\langle t\right\rangle}=\mathrm{d}\bar{c}^{\left\langle t\right\rangle}+\mathrm{d}\hat{c}^{\left\langle t\right\rangle}$

$\mathrm{d}\hat{c}^{\left\langle t\right\rangle}=\mathrm{d}a^{\left\langle t\right\rangle}*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\tanh（c^{\left\langle t\right\rangle}）^2)$

则有：
$$
\mathrm{d}{c}^{\left\langle t\right\rangle}=\mathrm{d}\bar{c}^{\left\langle t\right\rangle}+\mathrm{d}a^{\left\langle t\right\rangle}*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\tanh（c^{\left\langle t\right\rangle}）^2)
$$


$\mathrm{d}\Gamma_{o}^{\left\langle t\right\rangle} = \mathrm{d}a^{\left\langle t\right\rangle}* \tanh(c^{\left\langle t\right\rangle})$ 

$\mathrm{d}\tilde{c}^{\left\langle t\right\rangle}=\mathrm{d}c^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}$

$\mathrm{d}\Gamma_{f}^{\left\langle t\right\rangle} = \mathrm{d}c^{\left\langle t\right\rangle}*c^{\left\langle t-1\right\rangle}$

$\mathrm{d}\Gamma_{u}^{\left\langle t\right\rangle} = \mathrm{d}c^{\left\langle t\right\rangle}*\tilde{c}^{\left\langle t\right\rangle}$



计算参数梯度

$\begin{aligned}\mathrm{d}w_o &=\mathrm{d}\Gamma_{o}^{\left\langle t\right\rangle}*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\Gamma_{o}^{\left\langle t\right\rangle})* \left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T\\ &=\mathrm{d}a^{\left\langle t\right\rangle}* \tanh(c^{\left\langle t\right\rangle})*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\Gamma_{o}^{\left\langle t\right\rangle})* \left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T\end{aligned}$

$\begin{aligned}\mathrm{d}b_o &=\sum_{aixs=1}\mathrm{d}\Gamma_{o}^{\left\langle t\right\rangle}*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\Gamma_{o}^{\left\langle t\right\rangle})\\&=\sum_{aixs=1}\mathrm{d}a^{\left\langle t\right\rangle}* \tanh(c^{\left\langle t\right\rangle})*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\Gamma_{o}^{\left\langle t\right\rangle})\end{aligned}$

$\begin{aligned}\mathrm{d}w_f&=\mathrm{d}\Gamma_{f}^{\left\langle t\right\rangle}*\Gamma_{f}^{\left\langle t\right\rangle}*(1-\Gamma_{f}^{\left\langle t\right\rangle})*\left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T \\&=\mathrm{d}c^{\left\langle t\right\rangle}*c^{\left\langle t-1\right\rangle}*\Gamma_{f}^{\left\langle t\right\rangle}*(1-\Gamma_{f}^{\left\langle t\right\rangle})*\left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T \end{aligned}$

$\begin{aligned}\mathrm{d}b_f &=\sum_{aixs=1}\mathrm{d}\Gamma_{f}^{\left\langle t\right\rangle}*\Gamma_{f}^{\left\langle t\right\rangle}*(1-\Gamma_{f}^{\left\langle t\right\rangle})\\&=\sum_{aixs=1}\mathrm{d}c^{\left\langle t\right\rangle}*c^{\left\langle t-1\right\rangle}*\Gamma_{f}^{\left\langle t\right\rangle}*(1-\Gamma_{f}^{\left\langle t\right\rangle})\end{aligned}$

$\begin{aligned}\mathrm{d}w_u &=\mathrm{d}\Gamma_{u}^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-\Gamma_{u}^{\left\langle t\right\rangle})* \left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T\\ &=\mathrm{d}c^{\left\langle t\right\rangle}*\tilde{c}^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-\Gamma_{u}^{\left\langle t\right\rangle})* \left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T\end{aligned}$

$\begin{aligned}\mathrm{d}b_u &=\sum_{aixs=1}\mathrm{d}\Gamma_{u}^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-\Gamma_{u}^{\left\langle t\right\rangle})\\&=\sum_{aixs=1}\mathrm{d}c^{\left\langle t\right\rangle}*\tilde{c}^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-\Gamma_{u}^{\left\langle t\right\rangle})\end{aligned}$

$\begin{aligned}\mathrm{d}w_c &=\mathrm{d}\tilde{c}^{\left\langle t\right\rangle}*(1-(\tilde{c}^{\left\langle t\right\rangle})^2)* \left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T\\ &=\mathrm{d}c^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-(\tilde{c}^{\left\langle t\right\rangle})^2)* \left(\begin{array} {c}a^{\left \langle t-1\right\rangle} \\x^{\left \langle t\right\rangle}\end{array}\right)^T\end{aligned}$

$\begin{aligned}\mathrm{d}b_c &=\sum_{aixs=1}\mathrm{d}\tilde{c}^{\left\langle t\right\rangle}*(1-(\tilde{c}^{\left\langle t\right\rangle})^2)\\&=\sum_{aixs=1}\mathrm{d}c^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-（\tilde{c}^{\left\langle t\right\rangle})^2)\end{aligned}$



计算关于隐藏状态、先前记忆状态和输入的导数

$\mathrm{d}c^{\left\langle t-1\right\rangle}=\mathrm{d}c^{\left\langle t\right\rangle}*\Gamma_{f}^{\left\langle t\right\rangle}$

$\begin{aligned}\mathrm{d} a^{\left\langle t-1\right\rangle}&= \mathrm{d}\Gamma_{o}^{\left\langle t\right\rangle}*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\Gamma_{o}^{\left\langle t\right\rangle})*\hat{w}_o^T + \mathrm{d}\Gamma_{f}^{\left\langle t\right\rangle}*\Gamma_{f}^{\left\langle t\right\rangle}*(1-\Gamma_{f}^{\left\langle t\right\rangle})*\hat{w}_f^T \\&+\mathrm{d}\Gamma_{u}^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-\Gamma_{u}^{\left\langle t\right\rangle})*\hat{w}_u^T+\mathrm{d}\tilde{c}^{\left\langle t\right\rangle}*(1-(\tilde{c}^{\left\langle t\right\rangle})^2)*\hat{w}_c^T\end{aligned}$

$\begin{aligned}\mathrm{d} x^{\left\langle t\right\rangle}&= \mathrm{d}\Gamma_{o}^{\left\langle t\right\rangle}*\Gamma_{o}^{\left\langle t\right\rangle}*(1-\Gamma_{o}^{\left\langle t\right\rangle})*\tilde{w}_o^T + \mathrm{d}\Gamma_{f}^{\left\langle t\right\rangle}*\Gamma_{f}^{\left\langle t\right\rangle}*(1-\Gamma_{f}^{\left\langle t\right\rangle})*\tilde{w}_f^T \\&+\mathrm{d}\Gamma_{u}^{\left\langle t\right\rangle}*\Gamma_{u}^{\left\langle t\right\rangle}*(1-\Gamma_{u}^{\left\langle t\right\rangle})*\tilde{w}_u^T+\mathrm{d}\tilde{c}^{\left\langle t\right\rangle}*(1-(\tilde{c}^{\left\langle t\right\rangle})^2)*\tilde{w}_c^T\end{aligned}$

Where $\hat{w}^T$ denote $w[:, :n_a]$, $\tilde{w}^T$ denote $w[:, n_a:n_a+n_x]$

