
Pour répondre à ta première question importante : oui c'est bien ça.

Vérification 1 :
florian@florian-Precision-3450:~/Documents/expe_transfer_learning$ vim --version | grep clipboard
florian@florian-Precision-3450:~/Documents/expe_transfer_learning$

Vérification 2 :
florian@florian-Precision-3450:~/Documents/expe_transfer_learning$ nvim --version
NVIM v0.6.1
Build type: Release
LuaJIT 2.1.0-beta3
Compiled by team+vim@tracker.debian.org

Features: +acl +iconv +tui
See ":help feature-compile"

   system vimrc file: "$VIM/sysinit.vim"
  fall-back for $VIM: "/usr/share/nvim"

Run :checkhealth for more info
florian@florian-Precision-3450:~/Documents/expe_transfer_learning$

Vérification 3 :
florian@florian-Precision-3450:~/Documents/expe_transfer_learning$ echo $TERM
screen
florian@florian-Precision-3450:~/Documents/expe_transfer_learning$ tmux -V
tmux 3.2a

Et ce que tu demandais en plus à la fin :
florian@florian-Precision-3450:~/Documents/expe_transfer_learning$ echo $SSH_CONNECTION
10.10.8.50 43820 10.50.13.213 22
