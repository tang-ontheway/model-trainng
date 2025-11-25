#生成旋转矩阵
def pre_freqs_cis(dim: int, seq_len: int, theta: float = 10000.0):
    # 算的是θ,在d维度上进行以2为步长
    freqs = 1.0 / (theta ** (torch.arrange(0,dim,2)[: (dim // 2)].float() / dim))
    # 算的是m,这里的是t,标注的是token的位置
    t = torch.arrange(seq_len, device = freqs.device)
    # outer就是让前者的每一个乘上后者的每一位 结果是(len, dim //2)
    freqs = torch.outer(t, freqs).float()
    # torch.polar(magnitude, angle) m乘(cos,i*sin) 这里是矩阵版的1*cosmθ
    freq_cis = torch.polar(torch.ones_like(freqs), freqs)
    return freqs_cis

def apply_rotary(
    xq : torch.Tensor,
    xk : torch.Tensor,
    fre_cis : torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    #上述的代码是用来提示 输入参数的类型 和要返回的二元组 
    # xq.shape = [batch_size, seq_len, dim]
    # xq_.shape = [batch_size, seq_len, dim//2, 2]
    # *xq.shape[:-1]把除了最后一个维度dim的维度粘贴过来
    # (-1,2)起到的作用是把dim拆成两份 2表示拆成两份 -1表示自动填补维度
    xq_ = xq.float().reshape(*xq.shape[:-1], -1, 2)
    xk_ = xk.float().reshape(*xq.shape[:-1], -1, 2)

    #前面已经拆成了两个维度，现在直接转为复数
    xq_ = torch.view_as_complex(xq_)
    xk_ = torch.view_as_complex(xk_)

    #开始旋转,并展平为实数
    # xq_.shape = [batch_size, seq_len, dim//2, 2] freqs.shape = [seq_len, dim // 2] 
    # xq_out.shape = [batch_size, seq_len, dim]
    xq_out = torch.view_as_real(xq_ * freqs_cis).flatten(2)
    xk_out = torch.view_as_real(xk_ * freqs_cis).flatten(2)
    return xq_out.type_as(xq), xk_out.type_as(xk)
    


class Attention(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()

        self.wq = linear(...) #这里线性变换层的参数省略了
        self.wk = Linear(...)
        self.wv = Linear(...)

        self.freqs_cis = pre_freqs_cis(dim, max_seq_len * 2)
        # 为什么是最大len的二倍呢，因为rope说的是相对位置m-n,[-(L-1), (L-1)]这就是二倍了

    def forward(self, x: torch.Tensor):
        bsz , seqlen, _ = x.shape
        #分布乘上wq, wk,wv
        xq, xk, xv = self.wq(x), self.wk(x), self.wv(x)
        
        xq = xq.view(batch_size, seq_len, dim)
        xk = xk.view(batch_size, seq_len, dim)
        xv = xv.view(batch_size, seq_len, dim)

        #attention 操作之前,应用旋转
        xq, xk = apply_rotary(xq, xk, freqs_cis=freqs_cis)

        #matmul是矩阵乘法 xk.transpose(1, 2)是进行转置
        scores = torch.matmul(xq, xk.transpose(1, 2)) / math.sqrt(dim)
        # 对最后一个维度进行softmax, 每个i对j的注意力
        scores = F.softmax(scores.float(), dim = -1)
        output = torch.matmul(scores, xv)
        