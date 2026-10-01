(() => {
	"use strict";

	const byId = (id) => document.getElementById(id);
	const state = { rated: new Map(), searchMovie: null, similarMovie: null };

	function showError(message) {
		const error = byId("globalError");
		error.textContent = message;
		error.classList.remove("d-none");
	}

	function clearError() {
		const error = byId("globalError");
		error.textContent = "";
		error.classList.add("d-none");
	}

	function setLoading(active, message = "Working...") {
		const status = byId("globalStatus");
		status.replaceChildren();
		if (!active) return;
		const wrapper = document.createElement("span");
		wrapper.className = "loading-indicator";
		const spinner = document.createElement("span");
		spinner.className = "spinner-border spinner-border-sm";
		spinner.setAttribute("aria-hidden", "true");
		const text = document.createElement("span");
		text.textContent = message;
		wrapper.append(spinner, text);
		status.append(wrapper);
	}

	async function api(url, options = {}) {
		clearError();
		setLoading(true);
		try {
			const response = await fetch(url, options);
			const payload = await response.json();
			if (!response.ok) throw new Error(payload.error || `Request failed (${response.status})`);
			return payload;
		} catch (error) {
			showError(error.message || "The request failed. Please try again.");
			throw error;
		} finally {
			setLoading(false);
		}
	}

	function emptyState(message) {
		const node = document.createElement("p");
		node.className = "empty-state";
		node.textContent = message;
		return node;
	}

	function movieCard(movie, actionLabel, onAction) {
		const card = document.createElement("article");
		card.className = "movie-card";
		const body = document.createElement("div");
		const title = document.createElement("h3");
		title.className = "movie-title";
		title.textContent = movie.title;
		const meta = document.createElement("p");
		meta.className = "movie-meta";
		const year = movie.year ? String(movie.year) : "Year unknown";
		const genres = movie.genres || "Genre unavailable";
		const mean = movie.mean_rating == null ? "No ratings" : `Mean ${Number(movie.mean_rating).toFixed(2)}`;
		meta.textContent = `${year} · ${genres} · ${mean}`;
		body.append(title, meta);

		const footer = document.createElement("div");
		footer.className = "movie-foot";
		if (movie.score != null) {
			const badge = document.createElement("span");
			badge.className = "score-badge";
			badge.textContent = `Score ${Number(movie.score).toFixed(3)}`;
			footer.append(badge);
		} else {
			footer.append(document.createElement("span"));
		}
		if (actionLabel && onAction) {
			const action = document.createElement("button");
			action.type = "button";
			action.className = "btn btn-sm btn-outline-secondary";
			action.textContent = actionLabel;
			action.addEventListener("click", () => onAction(movie));
			footer.append(action);
		}
		card.append(body, footer);
		return card;
	}

	function renderCards(container, cards, emptyMessage, actionLabel, onAction) {
		container.replaceChildren();
		if (!cards.length) {
			container.append(emptyState(emptyMessage));
			return;
		}
		for (const movie of cards) container.append(movieCard(movie, actionLabel, onAction));
	}

	function showSelected(container, movie) {
		container.replaceChildren();
		const selected = document.createElement("div");
		selected.className = "selected-card";
		const title = document.createElement("h3");
		title.className = "movie-title";
		title.textContent = movie.title;
		const meta = document.createElement("p");
		meta.className = "movie-meta mb-0";
		meta.textContent = `${movie.year || "Year unknown"} · ${movie.genres || "Genre unavailable"}`;
		selected.append(title, meta);
		container.append(selected);
	}

	function attachAutocomplete(inputId, suggestionsId, onSelect) {
		const input = byId(inputId);
		const suggestions = byId(suggestionsId);
		let timer = null;
		let controller = null;

		input.addEventListener("input", () => {
			window.clearTimeout(timer);
			const query = input.value.trim();
			if (query.length < 2) {
				suggestions.replaceChildren();
				suggestions.classList.add("d-none");
				return;
			}
			timer = window.setTimeout(async () => {
				if (controller) controller.abort();
				controller = new AbortController();
				try {
					const response = await fetch(`/api/search?q=${encodeURIComponent(query)}`, { signal: controller.signal });
					const movies = await response.json();
					if (!response.ok) throw new Error(movies.error || "Search failed");
					suggestions.replaceChildren();
					if (!movies.length) {
						suggestions.append(emptyState("No matching titles."));
					} else {
						for (const movie of movies) {
							const button = document.createElement("button");
							button.type = "button";
							button.className = "suggestion";
							const title = document.createElement("span");
							title.className = "suggestion-title";
							title.textContent = `${movie.title}${movie.year ? ` (${movie.year})` : ""}`;
							const meta = document.createElement("span");
							meta.className = "suggestion-meta";
							meta.textContent = `${movie.genres || "Genre unavailable"} · ${movie.n_ratings} ratings`;
							button.append(title, meta);
							button.addEventListener("click", () => {
								input.value = movie.title;
								suggestions.replaceChildren();
								suggestions.classList.add("d-none");
								onSelect(movie);
							});
							suggestions.append(button);
						}
					}
					suggestions.classList.remove("d-none");
				} catch (error) {
					if (error.name !== "AbortError") showError(error.message);
				}
			}, 250);
		});

		document.addEventListener("click", (event) => {
			if (!suggestions.contains(event.target) && event.target !== input) suggestions.classList.add("d-none");
		});
	}

	attachAutocomplete("searchInput", "searchSuggestions", (movie) => {
		state.searchMovie = movie;
		showSelected(byId("selectedMovie"), movie);
		renderCards(byId("searchResults"), [], "Choose Find similar to explore this title.");
		const button = document.createElement("button");
		button.type = "button";
		button.className = "btn btn-dark mt-3";
		button.textContent = "Find similar";
		button.addEventListener("click", async () => {
			try {
				const cards = await api(`/api/similar/${movie.movie_id}?n=10`);
				renderCards(byId("searchResults"), cards, "No similar titles found.");
			} catch (_) { /* api() displays the message */ }
		});
		byId("selectedMovie").append(button);
	});

	attachAutocomplete("similarInput", "similarSuggestions", (movie) => {
		state.similarMovie = movie;
		showSelected(byId("similarSelected"), movie);
		byId("similarButton").disabled = false;
	});

	byId("similarButton").addEventListener("click", async () => {
		if (!state.similarMovie) return;
		try {
			const cards = await api(`/api/similar/${state.similarMovie.movie_id}?n=10`);
			renderCards(byId("similarResults"), cards, "No similar titles found.");
		} catch (_) { /* api() displays the message */ }
	});

	byId("userRecommendButton").addEventListener("click", async () => {
		const userId = Number(byId("userIdInput").value);
		try {
			const cards = await api(`/api/recommend/user/${encodeURIComponent(userId)}?n=10`);
			renderCards(byId("userResults"), cards, "No recommendations available.");
		} catch (_) { /* api() displays the message */ }
	});

	function refreshRatingControls() {
		byId("ratedCount").textContent = String(state.rated.size);
		byId("newRecommendButton").disabled = state.rated.size < 5;
	}

	function addRating(movie) {
		if (state.rated.has(movie.movie_id)) return;
		state.rated.set(movie.movie_id, { movie, rating: 0 });
		renderRatingList();
	}

	function renderRatingList() {
		const container = byId("ratingList");
		container.replaceChildren();
		for (const [movieId, entry] of state.rated.entries()) {
			const row = document.createElement("div");
			row.className = "rating-card";
			const title = document.createElement("strong");
			title.textContent = entry.movie.title;
			row.append(title);
			const controls = document.createElement("div");
			controls.className = "star-control";
			controls.setAttribute("role", "group");
			controls.setAttribute("aria-label", `Rate ${entry.movie.title}`);
			for (let value = 1; value <= 5; value += 1) {
				const star = document.createElement("button");
				star.type = "button";
				star.className = `star-button${value <= entry.rating ? " is-filled" : ""}`;
				star.textContent = "★";
				star.setAttribute("aria-label", `${value} out of 5`);
				star.setAttribute("aria-pressed", String(value === entry.rating));
				star.addEventListener("click", () => {
					state.rated.set(movieId, { movie: entry.movie, rating: value });
					renderRatingList();
				});
				controls.append(star);
			}
			const remove = document.createElement("button");
			remove.type = "button";
			remove.className = "btn btn-sm btn-outline-secondary ms-2";
			remove.textContent = "Remove";
			remove.addEventListener("click", () => {
				state.rated.delete(movieId);
				renderRatingList();
			});
			controls.append(remove);
			row.append(controls);
			container.append(row);
		}
		refreshRatingControls();
	}

	attachAutocomplete("rateSearchInput", "rateSuggestions", addRating);

	byId("newRecommendButton").addEventListener("click", async () => {
		const ratings = Array.from(state.rated.values())
			.filter((entry) => entry.rating >= 1 && entry.rating <= 5)
			.map((entry) => ({ movie_id: entry.movie.movie_id, rating: entry.rating }));
		try {
			const cards = await api("/api/recommend/new?n=12", {
				method: "POST",
				headers: { "Content-Type": "application/json" },
				body: JSON.stringify({ ratings }),
			});
			renderCards(byId("newUserResults"), cards, "No recommendations available.");
		} catch (_) { /* api() displays the message */ }
	});

	async function loadPopular() {
		try {
			const cards = await api("/api/popular?n=8");
			renderCards(byId("popularResults"), cards, "Popular titles could not be loaded.", "Rate", addRating);
		} catch (_) { /* api() displays the message */ }
	}

	loadPopular();
})();
