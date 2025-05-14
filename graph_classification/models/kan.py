import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch_scatter import scatter_add

class KANLinear(torch.nn.Module):
    def __init__(
        self,
        in_features,
        out_features,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        enable_standalone_scale_spline=True,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        # 增加动态网格支持
        dynamic_grid=False,
        max_degree=100,
    ):
        super(KANLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        # 动态网格支持
        self.dynamic_grid = dynamic_grid
        self.max_degree = max_degree  # 预设最大度数，用于归一化
        self.input_norm = nn.LayerNorm(in_features)
        # 新增参数
        self.update_counter = 0
        self.update_freq = 5  # 初始更新频率
        self.ema_alpha = 0.9  # EMA平滑系数
        if dynamic_grid:
            # 动态网格参数：度数映射到网格密度
            self.grid_size_mapper = torch.nn.Sequential(
                torch.nn.Linear(1, 16),
                torch.nn.ReLU(),
                torch.nn.Linear(16, 1),
                torch.nn.Sigmoid()
            )

        h = (grid_range[1] - grid_range[0]) / grid_size
        grid_points = torch.linspace(
            grid_range[0] - spline_order * h,
            grid_range[1] + spline_order * h,
            grid_size + 2 * spline_order + 1,
        )
        self.register_buffer("grid", grid_points.expand(in_features, -1).contiguous())

        self.base_weight = nn.Parameter(
            torch.Tensor(out_features, in_features)  # 维度需与线性层匹配
        )
        self.spline_weight = torch.nn.Parameter(
            torch.Tensor(in_features, grid_size + spline_order, out_features)
        )
        if enable_standalone_scale_spline:
            self.spline_scaler = nn.Parameter(
                torch.Tensor(in_features, out_features)
            )

        self.scale_noise = scale_noise
        self.scale_base = scale_base
        self.scale_spline = scale_spline
        self.enable_standalone_scale_spline = enable_standalone_scale_spline
        self.base_activation = base_activation()
        self.grid_eps = grid_eps
        self.update_freq = 5

        self.reset_parameters()

    def get_dynamic_grid_size(self, degrees):
        """ 动态调整网格大小 """
        normalized_deg = degrees.float() / self.max_degree
        grid_ratio = self.grid_size_mapper(normalized_deg.unsqueeze(-1))
        return 3 + (grid_ratio * 7).long()

    @property
    def scaled_spline_weight(self):
        # 修正缩放维度计算
        if self.enable_standalone_scale_spline:
            return self.spline_weight * self.spline_scaler.unsqueeze(1)  # (in, out) -> (in, 1, out)
        else:
            return self.spline_weight

    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5) * self.scale_base)
        with torch.no_grad():
            grid_center = self.grid.T[self.spline_order : self.spline_order + self.grid_size]  # (grid_size, in_features)
            noise = (
                    (torch.rand(
                        self.grid_size,  # 确保与grid_center维度一致
                        self.in_features,
                        self.out_features
                    ) - 0.5
                     ) * self.scale_noise
            )
            initial_coeff = self.curve2coeff(grid_center, noise)
            self.spline_weight.data.copy_(initial_coeff)  # (in, grid, out)

        if self.enable_standalone_scale_spline:
            torch.nn.init.kaiming_uniform_(
                self.spline_scaler,
                a=math.sqrt(5) * self.scale_spline
            )


    def b_splines(self, x: torch.Tensor):
        """
        Compute the B-spline bases for the given input tensor.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).

        Returns:
            torch.Tensor: B-spline bases tensor of shape (batch_size, in_features, grid_size + spline_order).
        """
        assert x.dim() == 2 and x.size(1) == self.in_features

        grid: torch.Tensor = (
            self.grid
        )  # (in_features, grid_size + 2 * spline_order + 1)
        x = x.unsqueeze(-1)
        bases = ((x >= grid[:, :-1]) & (x < grid[:, 1:])).to(x.dtype)
        for k in range(1, self.spline_order + 1):
            bases = (
                (x - grid[:, : -(k + 1)])
                / (grid[:, k:-1] - grid[:, : -(k + 1)])
                * bases[:, :, :-1]
            ) + (
                (grid[:, k + 1 :] - x)
                / (grid[:, k + 1 :] - grid[:, 1:(-k)])
                * bases[:, :, 1:]
            )

        assert bases.size() == (
            x.size(0),
            self.in_features,
            self.grid_size + self.spline_order,
        )
        return bases.contiguous()

    def curve2coeff(self, x: torch.Tensor, y: torch.Tensor):
        """
        Compute the coefficients of the curve that interpolates the given points.

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, in_features).
            y (torch.Tensor): Output tensor of shape (batch_size, in_features, out_features).

        Returns:
            torch.Tensor: Coefficients tensor of shape (out_features, in_features, grid_size + spline_order).
        """
        assert x.shape[0] == y.shape[0], f"Batch mismatch: x {x.shape} vs y {y.shape}"
        assert x.shape[1] == self.in_features, "Input feature dimension mismatch"
        assert y.shape[2] == self.out_features, "Output feature dimension mismatch"
        assert x.dim() == 2, "Input x must be 2D tensor"
        assert y.dim() == 3, "Input y must be 3D tensor"
        assert x.size(0) == y.size(0), f"Batch size mismatch: x {x.size(0)} vs y {y.size(0)}"
        assert x.size(1) == self.in_features, f"Input feature mismatch: {x.size(1)} vs {self.in_features}"
        assert y.size(2) == self.out_features, f"Output feature mismatch: {y.size(2)} vs {self.out_features}"

        assert x.shape == (self.grid_size, self.in_features), \
            f"输入维度错误: 期望({self.grid_size}, {self.in_features}) 实际{x.shape}"
        assert y.shape == (self.grid_size, self.in_features, self.out_features), \
            f"噪声维度错误: 期望({self.grid_size}, {self.in_features}, {self.out_features})"
        # 修改矩阵构造逻辑
        A = self.b_splines(x)  # (batch, in, coeff)
        batch_size = A.size(0)

        # 构造正则化项
        reg_matrix = torch.eye(A.size(-1), device=A.device) * 1e-6
        reg_matrix = reg_matrix.unsqueeze(0).expand(self.in_features, -1, -1)

        # 构造扩展矩阵
        A_ext = A.permute(1, 0, 2)  # (in, batch, coeff)
        A_reg = torch.cat([A_ext, reg_matrix], dim=1)  # (in, batch + coeff, coeff)

        # 构造目标矩阵
        y_ext = y.permute(1, 0, 2)  # (in, batch, out)
        y_reg = torch.cat([
            y_ext,
            torch.zeros(self.in_features, A.size(-1), self.out_features, device=y.device)
        ], dim=1)

        # 求解线性方程组
        solution = torch.linalg.lstsq(
            A_reg.reshape(self.in_features, -1, A.size(-1)),
            y_reg.reshape(self.in_features, -1, self.out_features)
        ).solution

        # 调整reshape逻辑
        coeff = solution.reshape(
            self.in_features,
            A.size(-1),  # grid_size + spline_order
            self.out_features
        )
        return solution

    def forward(self, x: torch.Tensor):
        batch_size = x.size(0)

        # 基础路径计算
        base_output = F.linear(
            self.base_activation(x),
            self.base_weight  # (out, in) -> (in, out)
        )

        # 样条路径计算
        spline_basis = self.b_splines(x)  # (batch, in, grid+order)
        spline_basis = spline_basis.view(batch_size, -1)  # (batch, in*(grid+order))

        # 调整权重视图
        spline_weight_view = self.scaled_spline_weight.reshape(
            -1,  # in*(grid+order)
            self.out_features
        ).T  # 转置为(out, in*(grid+order))

        spline_output = F.linear(
            spline_basis,
            spline_weight_view  # (out, in*(grid+order))
        )

        return base_output + spline_output

    @torch.no_grad()
    def update_grid(self, x: torch.Tensor, margin=0.01, epoch=None):
        if x.size(0) == 0:
            return

        # current_freq = self.update_freq
        # if epoch is not None:
        #     current_freq = max(3, min(10, self.update_freq - epoch // 10))
        #
        # self.update_counter += 1
        # if self.update_counter % current_freq != 0:
        #     return

        assert x.size(1) == self.in_features, \
            f"Input feature mismatch: Expected {self.in_features}, got {x.size(1)}"
        x = x.detach()  # 避免影响计算图
        x = self.input_norm(x)  # 应用LayerNorm

        # 分位数网格生成
        quantiles = torch.linspace(0, 1, self.grid_size + 1, device=x.device)
        grid_adaptive = torch.quantile(x, quantiles, dim=0).T  # (grid_size+1, in_features)

        # EMA平滑
        if hasattr(self, 'prev_grid'):
            grid_adaptive = self.ema_alpha * self.prev_grid + (1 - self.ema_alpha) * grid_adaptive
        self.prev_grid = grid_adaptive.clone()

        assert x.dim() == 2 and x.size(1) == self.in_features
        x_sorted = torch.sort(x, dim=0)[0]  # (batch, in_features)

        sample_indices = torch.linspace(0, x.size(0) - 1, self.grid_size + 1, device=x.device).long()
        grid_adaptive = x_sorted[sample_indices]  # (grid_size+1, in_features)

        feature_min = x_sorted[0]  # (in_features,)
        feature_max = x_sorted[-1]  # (in_features,)
        uniform_step = (feature_max - feature_min + 2 * margin) / self.grid_size  # (in_features,)

        arange_grid = torch.arange(self.grid_size + 1, device=x.device).float()  # (grid_size+1,)
        grid_uniform = (
                (feature_min - margin).unsqueeze(0) +  # (1, in_features)
                arange_grid.unsqueeze(-1) * uniform_step.unsqueeze(0)  # (grid_size+1, in_features)
        )

        # 混合自适应与均匀网格
        grid = self.grid_eps * grid_uniform + (1 - self.grid_eps) * grid_adaptive  # (grid_size+1, in_features)

        # 构建完整网格（包含扩展区域）
        full_grid = torch.cat([
            grid[:1] - (uniform_step.unsqueeze(0) * torch.arange(self.spline_order, 0, -1, device=x.device).unsqueeze(
                1)),
            grid,
            grid[-1:] + (uniform_step.unsqueeze(0) * torch.arange(1, self.spline_order + 1, device=x.device).unsqueeze(
                1))
        ], dim=0).T  # (in_features, grid_size + 2*spline_order + 1)

        self.grid.copy_(full_grid)

        #
        # splines = self.b_splines(x)  # (batch, in, coeff)
        # splines = splines.permute(1, 0, 2)  # (in, batch, coeff)
        # orig_coeff = self.scaled_spline_weight  # (out, in, coeff)
        # # orig_coeff = orig_coeff.permute(1, 2, 0)  # (in, coeff, out)
        # unreduced_spline_output = torch.bmm(splines, orig_coeff)  # (in, batch, out)
        # unreduced_spline_output = unreduced_spline_output.permute(
        #     1, 0, 2
        # )  # (batch, in, out)
        #
        # assert full_grid.size(0) == self.grid.size(0), \
        #     f"Grid size changed from {self.grid.size(0)} to {full_grid.size(0)}"
        #
        # self.grid.copy_(full_grid.T)  # 使用正确的变量名 grid
        #
        # x = torch.clamp(x, min=-5.0, max=5.0)
        # x = (x - x.mean(dim=0)) / (x.std(dim=0) + 1e-6)
        #
        # if x.size(0) < 10:  # 小批量或低方差时不更新
        #     return
        #
        # self.grid.copy_(grid.T)
        # self.spline_weight.data.copy_(self.curve2coeff(x, unreduced_spline_output))


    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        """
        Compute the regularization loss.

        This is a dumb simulation of the original L1 regularization as stated in the
        paper, since the original one requires computing absolutes and entropy from the
        expanded (batch, in_features, out_features) intermediate tensor, which is hidden
        behind the F.linear function if we want an memory efficient implementation.

        The L1 regularization is now computed as mean absolute value of the spline
        weights. The authors implementation also includes this term in addition to the
        sample-based regularization.
        """
        l1_fake = self.spline_weight.abs().mean(-1)
        regularization_loss_activation = l1_fake.sum()
        p = l1_fake / regularization_loss_activation
        regularization_loss_entropy = -torch.sum(p * p.log())
        return (
            regularize_activation * regularization_loss_activation
            + regularize_entropy * regularization_loss_entropy
        )


class KAN(torch.nn.Module):
    def __init__(
        self,
        layers_hidden,
        grid_size=5,
        spline_order=3,
        scale_noise=0.1,
        scale_base=1.0,
        scale_spline=1.0,
        base_activation=torch.nn.SiLU,
        grid_eps=0.02,
        grid_range=[-1, 1],
        dynamic_grid=False,  # 添加dynamic_grid参数
        max_degree=100,  # 添加max_degree参数
    ):
        super(KAN, self).__init__()
        self.grid_size = grid_size
        self.spline_order = spline_order
        # 添加dynamic_grid, max_degree参数
        self.dynamic_grid = dynamic_grid
        self.max_degree = max_degree

        self.layers = torch.nn.ModuleList()
        for in_features, out_features in zip(layers_hidden, layers_hidden[1:]):
            self.layers.append(
                KANLinear(
                    in_features,
                    out_features,
                    grid_size=grid_size,
                    spline_order=spline_order,
                    scale_noise=scale_noise,
                    scale_base=scale_base,
                    scale_spline=scale_spline,
                    base_activation=base_activation,
                    grid_eps=grid_eps,
                    grid_range=grid_range,
                    # 添加dynamic_grid, max_degree参数
                    dynamic_grid=dynamic_grid,
                    max_degree=max_degree,
                )
            )

    def get_dynamic_grid_size(self, degrees):
        """ 代理调用第一个 KANLinear 层的动态网格方法 """
        return self.layers[0].get_dynamic_grid_size(degrees)

    def forward(self, x: torch.Tensor, update_grid=False):
        for layer in self.layers:
            if update_grid:
                layer.update_grid(x)
            x = layer(x)
        return x

    def regularization_loss(self, regularize_activation=1.0, regularize_entropy=1.0):
        return sum(
            layer.regularization_loss(regularize_activation, regularize_entropy)
            for layer in self.layers
        )