# Adapted from the implementation in https://arxiv.org/abs/2106.12672.

import torch


class GBSTLayer(torch.nn.Module):
    """Performs Charformer GBST on a sequence.

    Attributes:
        embed_dim: Embedding dimension.
        downsample_rate: Integer of how much to downsample by.
        max_subword_block_width: Integer of max block size to use for
            enumeration.
        block_attention: Whether to use block score calibration.
        conv_kernel_size: Integer of the size of the pre-GBST
            convolution kernel.
    """

    def __init__(
            self,
            embed_dim: int,
            downsample_rate: int = 2,
            max_subword_block_width: int = 4,
            block_attention: bool = False,
            conv_kernel_size: int = 5,
    ):
        super().__init__()
        self.downsample_rate = downsample_rate
        self.max_subword_block_width = max_subword_block_width
        self.conv_kernel_size = conv_kernel_size
        self.conv_layer = torch.nn.Conv1d(
            embed_dim, embed_dim, self.conv_kernel_size)
        self.block_attention = block_attention
        self.block_scoring_network = torch.nn.Linear(embed_dim, 1, bias=False)

    def forward(self, inputs):
        """Performs downsampling on the character-scale input representation.

        Args:
            inputs: float Tensor of shape [batch_size, embedding_size,
                seq_length].

        Returns:
            <float>[batch_size, embedding_size,
                seq_length / downsample_rate]. Downsampled sequences.
        """
        length = inputs.shape[-1]

        if self.conv_kernel_size:
            inputs = self.conv_layer(inputs)

        # TODO fix downsampling with causality; max(subblock_width, downsample_rate)?
        all_block_scores = []
        all_sequences = []
        for subword_len in range(1, self.max_subword_block_width):
            padded_input = inputs
            # Pad the sequence length if needed.
            if length % subword_len != 0:
                pad_amt = subword_len - int(length % subword_len)
                padding = (0, pad_amt, 0, 0, 0, 0)
                assert len(padding) == inputs.dim() * 2
                padded_input = torch.nn.functional.pad(inputs, padding)

            # For this block size, form candidate block embeddings and scores.
            # candidates shape: [batch, dim, seq_len/subword_len]
            # block_scores shape: [batch, 1, seq_len/subword_len]
            candidates = torch.nn.functional.avg_pool1d(
                padded_input, (subword_len,), stride=(subword_len,))
            block_scores = self.block_scoring_network(
                candidates.transpose(-2, -1),
            ).transpose(-2, -1)

            # Upsample it back to the original sequence length.
            retiled_seq = torch.repeat_interleave(
                candidates, subword_len, dim=-1)
            retiled_block_scores = torch.repeat_interleave(
                block_scores, subword_len, dim=-1)

            # Repad the upsampled sequence if needed.
            if retiled_block_scores.shape[-1] < length:
                repad_amt = length - retiled_block_scores.shape[-1]
                repadding = (0, repad_amt, 0, 0, 0, 0)
                assert (
                    len(repadding)
                    == retiled_seq.dim() * 2
                    == retiled_block_scores.dim() * 2
                )
                retiled_seq = torch.nn.functional.pad(retiled_seq, repadding)
                retiled_block_scores = torch.nn.functional.pad(
                    retiled_block_scores, repadding)

            # Make sure everything is the right length and add new
            # dimension to concat candidate blocks on.
            retiled_block_scores = retiled_block_scores[:, :, :length, None]
            retiled_seq = retiled_seq[:, :, :length, None]
            all_block_scores.append(retiled_block_scores)
            all_sequences.append(retiled_seq)

        # [batch, 1, length, num_candidates]
        block_scores = torch.cat(all_block_scores, dim=-1)
        if self.block_attention:
            att = block_scores @ block_scores.transpose(-1, -2)
            att = torch.nn.functional.softmax(att, dim=-1)
            block_scores = att @ block_scores
        else:
            block_scores = torch.nn.functional.softmax(block_scores, dim=-1)

        candidates = torch.cat(all_sequences, dim=-1)

        # [batch, dim, length, num_candidates]
        candidates = candidates * block_scores
        # [batch, dim, length]
        output = torch.sum(candidates, axis=-1)

        # Downsample by mean pooling.
        if self.downsample_rate > 1:
            output = torch.nn.functional.avg_pool1d(
                output,
                (self.downsample_rate,),
                stride=(self.downsample_rate,),
            )
        return output
