#!/usr/bin/env Rscript

# Simulation reproductible des deux jeux de données décrits dans protocole.md.

set.seed(20260716)

dir.create("data", showWarnings = FALSE)
dir.create("figures", showWarnings = FALSE)

n <- 1000L

construire_plan <- function() {
  cellules <- data.frame(
    X1 = rep(c("a1", "b1"), times = 3L),
    X2 = rep(c("a2", "b2", "c2"), each = 2L),
    effectif = c(167L, 167L, 167L, 167L, 166L, 166L)
  )

  plan <- cellules[rep(seq_len(nrow(cellules)), cellules$effectif), c("X1", "X2")]
  plan <- plan[sample.int(nrow(plan)), , drop = FALSE]
  rownames(plan) <- NULL
  plan$X1 <- factor(plan$X1, levels = c("a1", "b1"), labels = c("source", "cible"))
  plan$X2 <- factor(plan$X2, levels = c("a2", "b2", "c2"))
  plan$X3 <- rnorm(n)
  plan$X4 <- rnorm(n)
  plan
}

simuler_modele_1 <- function() {
  donnees <- construire_plan()

  mu <- 2
  beta3 <- 1.5
  sigma <- 1
  mu1 <- c(source = 0, cible = 1)
  mu2 <- c(a2 = 0, b2 = -0.75, c2 = 0.75)
  gamma1 <- c(source = 0, cible = -0.5)

  donnees$signal <- mu + mu1[as.character(donnees$X1)] +
    mu2[as.character(donnees$X2)] +
    (beta3 + gamma1[as.character(donnees$X1)]) * donnees$X3
  donnees$epsilon <- rnorm(n, mean = 0, sd = sigma)
  donnees$Y <- donnees$signal + donnees$epsilon
  donnees$modele <- "Modèle 1"
  donnees$id <- seq_len(n)
  donnees[, c("id", "modele", "X1", "X2", "X3", "X4", "signal", "epsilon", "Y")]
}

simuler_modele_2 <- function() {
  donnees <- construire_plan()

  mu <- 2
  beta3 <- 1.2
  beta34 <- 0.8
  sigma <- 1
  mu1 <- c(source = 0, cible = 0.75)
  gamma2 <- c(a2 = 0, b2 = 0.4, c2 = -0.3)
  mu12 <- matrix(
    c(0, -1, 1, 0, -0.5, 1.5),
    nrow = 2L, byrow = TRUE,
    dimnames = list(c("source", "cible"), c("a2", "b2", "c2"))
  )
  gamma12 <- matrix(
    c(0, 0.4, -0.4, -0.3, 0.7, -0.8),
    nrow = 2L, byrow = TRUE,
    dimnames = list(c("source", "cible"), c("a2", "b2", "c2"))
  )

  i <- cbind(as.character(donnees$X1), as.character(donnees$X2))
  donnees$signal <- mu + mu1[as.character(donnees$X1)] + mu12[i] +
    (beta3 + gamma12[i]) * donnees$X3 +
    (beta34 + gamma2[as.character(donnees$X2)]) * donnees$X3 * donnees$X4
  donnees$epsilon <- rnorm(n, mean = 0, sd = sigma)
  donnees$Y <- donnees$signal + donnees$epsilon
  donnees$modele <- "Modèle 2"
  donnees$id <- seq_len(n)
  donnees[, c("id", "modele", "X1", "X2", "X3", "X4", "signal", "epsilon", "Y")]
}

modele_1 <- simuler_modele_1()
modele_2 <- simuler_modele_2()

stopifnot(
  nrow(modele_1) == n,
  nrow(modele_2) == n,
  all(table(modele_1$X1) == 500L),
  all(table(modele_2$X1) == 500L),
  all(table(modele_1$X1, modele_1$X2) %in% c(166L, 167L)),
  all(table(modele_2$X1, modele_2$X2) %in% c(166L, 167L))
)

write.csv(modele_1, "data/donnees_simulees_modele_1.csv", row.names = FALSE)
write.csv(modele_2, "data/donnees_simulees_modele_2.csv", row.names = FALSE)

donnees <- rbind(modele_1, modele_2)
resume <- aggregate(
  Y ~ modele + X1,
  data = donnees,
  FUN = function(x) c(n = length(x), moyenne = mean(x), ecart_type = sd(x), mediane = median(x))
)
resume <- cbind(resume[c("modele", "X1")], as.data.frame(resume$Y))
write.csv(resume, "data/resume_simulations.csv", row.names = FALSE)

if (!requireNamespace("ggplot2", quietly = TRUE)) {
  stop("Le paquet ggplot2 est nécessaire pour générer la figure.")
}

p <- ggplot2::ggplot(donnees, ggplot2::aes(x = X1, y = Y, fill = X1)) +
  ggplot2::geom_violin(trim = FALSE, alpha = 0.65, color = NA) +
  ggplot2::geom_boxplot(width = 0.18, outlier.alpha = 0.25, fill = "white") +
  ggplot2::facet_wrap(~modele, scales = "free_y") +
  ggplot2::scale_fill_manual(values = c(source = "#4477AA", cible = "#CC6677")) +
  ggplot2::labs(
    title = "Distribution de la réponse simulée par domaine",
    subtitle = "1 000 observations par modèle, dont 500 par domaine",
    x = "Domaine (X1)", y = "Réponse Y"
  ) +
  ggplot2::theme_minimal(base_size = 12) +
  ggplot2::theme(
    legend.position = "none",
    panel.grid.minor = ggplot2::element_blank(),
    plot.title.position = "plot"
  )

ggplot2::ggsave("figures/distribution_Y_par_domaine.pdf", p, width = 8, height = 4.8)
ggplot2::ggsave("figures/distribution_Y_par_domaine.png", p, width = 8, height = 4.8, dpi = 300)

print(resume, row.names = FALSE)
