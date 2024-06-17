ESM_DIM = 1280
try:
    import esm
    import torch
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    
    esm2, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    esm2 = esm2.to(device)

    batch_converter = alphabet.get_batch_converter()
    batch_converter = batch_converter

    esm2.eval()

    def chain2esm(chain_name:str, sequence:str):

        """
        chain: H/L/A sequence, from selected_pepetides(H, L)
        
        """

        try:

            data = [(chain_name, sequence)]
            batch_labels, batch_strs, batch_tokens = batch_converter(data)
            batch_lens = (batch_tokens != alphabet.padding_idx).sum(1)
            tokens_len = batch_lens[0]
            batch_tokens = batch_tokens.to(device)
            with torch.no_grad():   
                results = esm2(batch_tokens, repr_layers=[33], return_contacts=True)    
            token_representations = results["representations"][33]
            ESM = token_representations[0, 1 : tokens_len - 1]
            return ESM.cpu().numpy()


        except Exception as e:

            print(f'{e}, {sequence}')

            return torch.zeros([len(sequence), ESM_DIM], dtype=torch.float32)
except:
    print('ESM not installed, please install it first')
    pass