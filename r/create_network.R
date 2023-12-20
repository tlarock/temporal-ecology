library(ggplot2)

# Create a seasonal network

# Sample mass train species
x <- seq(0, 1, length.out = 100)
y.beta <- dbeta(x, shape1 = 2, shape2 = 5)
plot(y ~ x, type = "l")

species_mass <- rbeta(25, 2, 5)

hist(species_mass)
sort(species_mass)

# Sample arrival date
y.norm <- dnorm(x, 0.25, 0.05)
plot(y.norm ~ x, type = "l")

arrival_date <- rnorm(25, 0.25, 0.05)

hist(arrival_date)
sort(arrival_date)

# Sample duration time
y.dur <- dnorm(x, 0.5, 0.1)
plot(y.dur ~ x, type = "l")

duration <- rnorm(25, 0.5, 0.1)

hist(duration)
sort(duration)

# Departure date
departue <- arrival_date + duration


hist(departue)
sort(departue)

# lolipop plot
df_dates <- tibble(
  species = seq(1:length(species_mass)), 
  # mass = species_mass,
  arrival = arrival_date,
  departue = departue
) %>% 
  pivot_longer(-species)

df_dates %>% 
  ggplot(aes(x = value, y = species)) +
  
  geom_line(aes(group=species), color="#E7E7E7", linewidth=3.5) + 
  # note that linewidth is a little larger than the point size 
  # so that the line matches the height of the point. why is it 
  # like that? i don't really know
  
  geom_point(aes(color=name), size=3) +
  theme_minimal() +
  theme(legend.position = "none",
        axis.text.y = element_text(color="black"),
        axis.text.x = element_text(color="#989898"),
        axis.title = element_blank(),
        panel.grid = element_blank()
  ) +
  scale_color_manual(values=c("#436685", "#BF2F24"))+
  scale_x_continuous(labels = scales::percent_format(scale = 1))

